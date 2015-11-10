import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib
from time import sleep
import re
import concurrent.futures

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def collect_links(gallery_page,links_filename):
    
    painting_links = []
    
    response = requests.get(gallery_page, stream=True)
    soup = BeautifulSoup(response.content, from_encoding='UTF-8')
    all_pages =[a_page['href'] for a_page in soup.select('#ulPaging .gener_links a')]
    
    for a_page in all_pages:
        response = requests.get(a_page, stream=True)
        soup = BeautifulSoup(response.content, from_encoding='UTF-8')
        painting_links.extend(a_painting['href'] for a_painting in soup.select('table.PdtSet .Attn a'))
        #sleep(3)
    with open(links_filename, 'w') as f:
        for a_painting_link in painting_links:
            #print a_painting_link
            f.write(a_painting_link+'\n')
    return ("MY_INFO: Collected links and written to " + links_filename,painting_links)




def scrape_painting_link(its_link,category):
    
    global paintings_df
    
    response = requests.get(its_link, stream=True)
    soup = BeautifulSoup(response.content, from_encoding='UTF-8')
        
    image_scrape_status,painting_id,paintings_df = scrape_image(soup)
    print image_scrape_status
    if re.match('MY_INFO: SKIP',image_scrape_status):
        old_category = paintings_df.loc[painting_id,'category']
        paintings_df.set_value(painting_id,'category', old_category +", "+category)
        return "MY_INFO: Parsing done, existing painting #",painting_id
    paintings_df.set_value(painting_id,'buy_link',its_link)
    paintings_df.set_value(painting_id,'category',category)
    description_scrape_status,paintings_df = scrape_descriptions(soup,painting_id)
    print description_scrape_status
    sleep(5)
    return "MY_INFO: Parsing done, painting # "+painting_id,paintings_df

def scrape_image(soup):
    global paintings_df
    
    jpeg_link = re.findall("http.*jpg",soup.select('.cloudzoom')[0]['data-cloudzoom'])[0]    
    painting_id = jpeg_link.split("/")[-3]
    jpeg_filename = painting_id + "__" + jpeg_link.split('/')[-1]
    if painting_id in paintings_df.index.values:
        return "MY_INFO: SKIP image# "+ str(painting_id), painting_id,paintings_df
    paintings_df.set_value(painting_id,'its_jpeg_link',jpeg_link)
    paintings_df.set_value(painting_id,'its_jpeg',jpeg_filename)
    urllib.urlretrieve(jpeg_link,jpeg_dir+jpeg_filename)
    return "MY_INFO: Saved JPEG for image# "+ str(painting_id), painting_id,paintings_df


def scrape_descriptions(soup,painting_id):
    global paintings_df
    
    title_price_etc = soup.select('.cloudzoom')[0]['alt'].split("|")[1:3]
    
    paintings_df.set_value(painting_id,'title',title_price_etc[0].split(" by ")[0])
    paintings_df.set_value(painting_id,'artist',title_price_etc[0].split(" by ")[1])
    paintings_df.set_value(painting_id,'price',title_price_etc[1])
      
    for a_descr,description in zip(['painting_descr','artist_bio'],soup.select('.art_header')):
        its_text = description.fetchNextSiblings()[0].get_text()
        its_text = re.sub('^\n','',its_text)
        paintings_df.set_value(painting_id,a_descr,its_text)
        
    return "MY_INFO: Scraped description for image #"+ str(painting_id),paintings_df


paintings_df = pd.DataFrame(columns=['title','artist','price',\
                                     'category', 'painting_descr','artist_bio',
                                     'its_jpeg_link','its_jpeg','buy_link'])

jpeg_dir = '../scrape_images/image_data/'     
#if __name__ == '__main__':
def main_run():
        styles = ['genre-abstract', 'classical-artwork', 'genre-expressionism',\
                  'genre-impressionism','minimalism-artwork','modern-artwork', \
                  'genre-pop culture','primitive-artwork', 'realism-artwork', \
                  'street-art-artwork', 'genre-surrealism', 'vintage-artwork']
        global paintings_df 
        for style in styles:
                links_filename = 'links_to_' + style + '.txt'
                with open(links_filename) as f:
                        painting_links = f.read().splitlines()
                print "MY_INFO: Scraping for ",style," paintings"
                for a_chunk in chunks(painting_links,len(painting_links)/5):
                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                future_to_url = {executor.submit(scrape_painting_link,url,style): url for url in a_chunk}
                                for future in concurrent.futures.as_completed(future_to_url):
                                        url = future_to_url[future]
                                        try:
                                                data = future.result()
                                        except Exception as exc:
                                                print('MY_ERROR: %r generated an exception: %s' % (url, exc))
                                        else:
                                                print('MY_INFO: %r page is %d bytes' % (url, len(data)))
                print "MY_INFO: Done scraping for ",style," paintings"
        paintings_df.to_pickle('paintingspickle.p')
                

#Already wrote links to file
#for style in styles:
#    gallery_page = 'http://www.ugallery.com/'+style+'/painting'
#    links_filename = 'links_to_' + style + '.txt'
#    collect_link_status, painting_links = collect_links(gallery_page,links_filename)
#    print collect_link_status
