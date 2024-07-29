import scrapy
from ..items import QAItem
from bs4 import BeautifulSoup 

class QASpider(scrapy.Spider):
    name = 'qa'
    page_number = 1
    base_url = 'https://thuvienphapluat.vn/hoi-dap-phap-luat/tien-te-ngan-hang?page={}'
    
    start_urls = [
        base_url.format(page_number)
    ]
    
    def remove_query_parameter(self, url):
        return url.split('&mode')[0]
    
    def parse(self, response):        
        all_cards = response.css(".news-card a")
        
        for card in all_cards:
            href = card.xpath("@href").extract_first()
            
            if href.startswith("https:"):
                yield response.follow(href, self.parse_document, meta={'download_timeout': 1})
            else:
                yield
                
        next_page = QASpider.base_url.format(QASpider.page_number + 1)
        
        if next_page is not None and QASpider.page_number < 1:
            QASpider.page_number += 1
            yield response.follow(next_page, callback=self.parse)
            
    def parse_document(self, response):
        
        items = QAItem()
        soup = BeautifulSoup(response.body, 'html5lib')
        
        questions = soup.find_all('h2')
        
        qa_pairs = []
        
        for question in questions:
            item = {"question": question.text}
            rel_docs = []
            
            # Find all <p> siblings following the <h2> tag
            sibling_p_tags = question.find_next_siblings('p')
            
            for p in sibling_p_tags:
                # Extract hrefs from <a> tags within <p> siblings
                a_tags = p.find_all('a')
                for a in a_tags:
                    title = a.text
                    href = self.remove_query_parameter(a.get('href'))
                    rel_docs.append({"title": title, "href": href})
                
            item['rel_docs'] = rel_docs
            qa_pairs.append(item)
        
        # Remove overlapping rel_docs:
        cummulative = 0
        for i in range(len(qa_pairs)-1, -1, -1):
            qa_pairs[i]['rel_docs'] = qa_pairs[i]['rel_docs'][:len(qa_pairs[i]['rel_docs']) - cummulative]
            cummulative += len(qa_pairs[i]['rel_docs'])
        
        items['qa_pairs'] = qa_pairs
        
        yield items
