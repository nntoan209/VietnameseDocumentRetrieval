# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class MetaDataItem(scrapy.Item):
    # define the fields for your item here like:
    id = scrapy.Field()
    title = scrapy.Field()
    href = scrapy.Field()
    so_hieu = scrapy.Field()
    loai_van_ban = scrapy.Field()
    noi_ban_hanh = scrapy.Field()
    tinh_trang = scrapy.Field()
    content = scrapy.Field()

class QAItem(scrapy.Item):
    qa_pairs = scrapy.Field()