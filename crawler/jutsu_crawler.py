import scrapy
from bs4 import BeautifulSoup

# Explore the naruto.fandom.wiki page and inspect elements for better understanding.
class BlogSpider(scrapy.Spider):
    name = 'narutospider'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

    # Parses the main page containing the jutsu lists and then moves to the next page containing next batch of jutsus.
    def parse(self, response):
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
            # Parsing the jutsu information from its individual information page.
            extracted_data = scrapy.Request("https://naruto.fandom.com"+href,
                           callback=self.parse_jutsu)
            
            # Jutsu info saved as jsonl file.
            yield extracted_data

        # Moving to next page
        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)
    
    def parse_jutsu(self, response):
        # Getting jutsu name
        jutsu_name = response.css("span.mw-page-title-main::text").extract()[0]
        jutsu_name = jutsu_name.strip()
        
        # Getting jutsu description and trimming blank spaces at the ends. 
        div_selector = response.css("div.mw-parser-output")[0]
        div_html = div_selector.extract()

        soup = BeautifulSoup(div_html).find('div')

        # Getting info about jutsu type if it exists.
        jutsu_type=""
        if soup.find('aside'):
            aside = soup.find('aside')

            for cell in aside.find_all('div',{'class':'pi-data'}):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == "Classification":
                        jutsu_type = cell.find('div').text.strip()

        soup.find('aside').decompose()

        # Removing Trivia section from description.
        jutsu_description = soup.text.strip()
        jutsu_description = jutsu_description.split('Trivia')[0].strip()

        return dict (
            jutsu_name = jutsu_name,
            jutsu_type = jutsu_type,
            jutsu_description = jutsu_description
        )
        
# To run this crawler:
# scrapy runspider crawler/jutsu_crawler.py -o data/jutsus.jsonl