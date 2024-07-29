import scrapy
from ..items import MetaDataItem
from bs4 import BeautifulSoup 

class LawSpider(scrapy.Spider):
    name = 'law'
    page_number = 1
    base_url = 'https://thuvienphapluat.vn/archive/Thong-tu/t23p{}.aspx'
    
    start_urls = [
        base_url.format(page_number)
    ]
    
    def parse(self, response):        
        all_laws = response.css("div#content ol li a")
        
        for law in all_laws:
            title = law.css("::text").extract_first()
            href = law.xpath("@href").extract_first()
            
            if "Circular" in title:
                self.logger.info("Skip: %s", title) 
                yield
            else:
                yield response.follow(href, self.parse_document, meta={'title': title, 'href': href, 'download_timeout': 60})
                
        next_page = LawSpider.base_url.format(LawSpider.page_number + 1)
        
        if next_page is not None and LawSpider.page_number < 1:
            LawSpider.page_number += 1
            yield response.follow(next_page, callback=self.parse)
            
    def parse_document(self, response):
        title = response.meta.get('title', '')
        href = response.meta.get('href', '')
        
        item = MetaDataItem()
        
        so_hieu = response.css("#divThuocTinh tr:nth-child(1) td:nth-child(2)::text").extract_first().strip()
        loai_van_ban = response.css("#divThuocTinh tr:nth-child(1) td:nth-child(5)::text").extract_first().strip()
        noi_ban_hanh = response.css("#divThuocTinh tr:nth-child(2) td:nth-child(2)::text").extract_first().strip()
        tinh_trang = response.css(".text-red span::text").extract_first()
        
        item['id'] = response.url.split('-')[-1].split('.')[0]
        item['title'] = title
        item['href'] = href
        item['so_hieu'] = so_hieu
        item['loai_van_ban'] = loai_van_ban
        item['noi_ban_hanh'] = noi_ban_hanh
        item['tinh_trang'] = tinh_trang
        
        soup = BeautifulSoup(response.body, 'html5lib')
        for p in soup.find_all('p'):
            for a in p.find_all('a', href=True):
                # Replace the <a> tag with its text content
                a.replace_with(a.get_text())
                
            for strong_tag in soup.find_all('strong'):
                span_tag = strong_tag.find('span')
                if span_tag:
                    # Replace <strong><span>Some text</span></strong> with Some text
                    strong_tag.replace_with(span_tag.get_text())
        
        first_a_tag = soup.find('a', {'name': 'chuong_1'})
        if not first_a_tag:
            first_a_tag = soup.find('a', {'name': 'dieu_1'})
        
        if not first_a_tag:
            yield
        else:
            item['content'] = [first_a_tag.text]
            # Get the parent <p> tag
            parent_p_tag = first_a_tag.find_parent('p')
            
            if parent_p_tag:
                # Get the next sibling <p> tag
                next_sibling_p_tag = parent_p_tag.find_next_siblings('p')
                for sibling in next_sibling_p_tag:
                    item['content'].append(sibling.text.replace('\n', ' ').strip())
            
                # Remove all empty string or spaces
                # if item['content']:
                #     item['content'] = [content for content in item['content'] if content not in ['', ' ']]
            yield item
