import os
import re
import csv
import sys
import yaml
import time
import shutil
import requests
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from typing import Iterable, DefaultDict
from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.remote.webdriver import Remote


"""        目,      [亚目],        [总科],        科,          [亚科],         [族],      属,      [亚属],       种
        Order,   Suborder,   Superfamily,    Family,     [Subfamily],     [Tribe],   Genus,   Subgenus,   Species.
            0           1              2          3               4             5        6           7          8
"""
_label_index = [('亚目', 1), ('总科', 2), ('亚科', 4), ('亚属', 7),  ('目', 0), ('科', 3), ('族', 5),  ('属', 6), ('种', 8)]
csv_index = '目 Order,亚目 Suborder,总科 Superfamily,科 Family,亚科 Subfamily,族 Tribe,属 Genus,亚属 Subgenus,种 Species,百科页面 WebPages,特征描述 Description,示例图像 Imgs'
item2idx = {item: idx for idx, item in enumerate(csv_index.split(','))}


global driver
global logfile
global text_img_pairs


def initialize_browser():
    os.system('start chrome --remote-debugging-port=9527 --incognito --disable-extensions')  # --blink-settings=imagesEnabled=false
    # os.system('start chrome --remote-debugging-port=9527 --incognito --blink-settings=imagesEnabled=false --disable-extensions')
    options = Options()
    # options.add_argument('--headless')
    # options.add_argument('--disable-gpu')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('debuggerAddress', '127.0.0.1:9527')
    
    global driver
    driver = webdriver.Chrome(options=options)
    
    # port = driver.service.port
    # remote_driver = Remote(command_executor='http://localhost:{}/wd/hub'.format(port), desired_capabilities={})
    
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                      get: () => undefined
                    })
                  """
    })
    
    # driver.get('https://bot.sannysoft.com/')
    # time.sleep(10)
    

def walks_in_Baidu_by_driver(name) -> [str, str, Iterable]:
    
    desc = ''
    webpage = ''
    img_urls = []  # format [(alt,url),(alt,url)]
    global logfile
    
    zh_name, latin_name = name.split(' ', 1)
    
    url = f'https://baike.baidu.com/item/{zh_name}'
    driver.execute_script('''window.location.href = '{}';'''.format(url))
    
    # if in multi-item page
    driver.execute_script('''
        next_page = ''
        document.querySelectorAll('.custom_dot.para-list.list-paddingleft-1 .list-dot.list-dot-paddingleft div.para')
        .forEach((item) => {
            if (['科', '属', '动物'].some(str => item.innerText.includes(str))){
                if (item.children[0]) {
                    next_page = item.children[0].href;
                    return item.children[0].href;
                }
            }
        })
        if(next_page) window.location.href = next_page;
    ''')
    
    # if page do not exist, check latin_name first
    driver.execute_script('''
        if(window.location.href.includes('error') || !window.location.href.includes('item'))
            window.location.href = 'https://baike.baidu.com/item/{}';
    '''.format(latin_name))
    
    # # then try latin name search
    # driver.execute_script('''
    #     if(window.location.href.includes('error') || !window.location.href.includes('item'))
    #         window.location.href = 'https://baike.baidu.com/search?word={}';
    # '''.format(latin_name))
    # # check first result
    # driver.execute_script('''
    #     first_result = document.querySelector('.result-title')
    #     if (first_result) window.location.href = first_result.href;
    # ''')
    
    # check page
    if (driver.execute_script('''
            return window.location.href.includes('item');
    ''') and driver.execute_script('''
            desc = document.querySelector('.lemma-desc');
            return (desc && ['科', '属', '动物'].some(str => desc.innerText.includes(str)) || ['{}', '{}', '动物'].some(str => document.documentElement.outerHTML.includes(str)));
    '''.format(zh_name, latin_name))) or driver.execute_script('''
            if(document.querySelector('.lemmaWgt-subLemmaListTitle')) return true;
            return false;
    '''):
        pass
    else:
        logfile.write(f'{zh_name} page not found\n')
        return webpage, desc, img_urls
    webpage = driver.execute_script('''return window.location.href;''')
    
    # get description
    # too redundancy to collect
    
    # get album page
    album_url = driver.execute_script('''
            more = document.querySelector('.more-link');
            return more == null ? '' : more.href;
    ''')
    if not album_url:
        logfile.write(f'{zh_name} album not found\n')
        return webpage, desc, img_urls
    driver.execute_script('''window.location.href = '{}';'''.format(album_url))
    driver.execute_script('''
        window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth',
        complete: () => {
                console.log('页面滚动完成');
            }
        });
    ''')
    time.sleep(3)
    
    # deal each album
    img_urls = driver.execute_script('''
            img_urls = [];
            albums = document.querySelectorAll('.pic-list');
            albums.forEach((album) => {
                for (let i = 1; i < album.children.length; i++) {
                    img = album.children[i].children[1];
                    if(['分布', '演变', '迁移', '画', '字', '过程', '示意'].some(str => img.alt.includes(str))) continue;
                    img_urls.push([img.alt ? img.alt : ''' + f'"{zh_name}"' + ''', img.src.split('?')[0]]);
                }
            });
            return img_urls;
    ''')

    assert len(img_urls)
    logfile.write(f'name {zh_name} find {len(img_urls):2d} imgs\n')
    return webpage, desc, img_urls


def walks_in_Baidu(zh_name) -> [str, str, Iterable]:
    
    desc = ''
    webpage = ''
    img_urls = []  # format [(alt,url),(alt,url)]
    
    url = f'https://baike.baidu.com/item/{requests.utils.quote(zh_name)}'
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
    response = requests.get(url, headers=headers)
    
    # if do not exist, return
    if 'https://baike.baidu.com/error.html' in response.text or zh_name not in response.text:
        print(f'{zh_name} page not found')
        return webpage, desc, img_urls
    webpage = url
    
    # get description
    # too redundancy to collect
    
    # get album page
    album_page = re.findall(r'<a class="more-link".*?href="(.*?)".*?>', response.text, re.S)
    if len(album_page) < 1:
        print(f'{zh_name} album not found')
        return webpage, desc, img_urls
    
    album_page = album_page[0]
    # deal each album
    url = f'https://baike.baidu.com{album_page}'
    # print(url)
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for album_tag in soup.select('div.pic-list'):
        for a_tag in album_tag.select('a')[1:]:        # remove repeated album cover
            img_tag = a_tag.select('img')[0]
            if any(keyword in img_tag.attrs['alt'] for keyword in ['分布', '演变', '迁移', '画', '字']):
                continue                               # skip no animal picture
            img_url = img_tag.attrs['src']
            img_urls.append(f'({img_tag.attrs["alt"]}, {img_url.split("?")[0]})')
    
    time.sleep(1.5)
    return webpage, desc, img_urls
    # return f"\"{','.join(imgs)}\""


def walks_in_Wiki(zh_name) -> [str, str, Iterable]:
    pass


def name_augment(name: str):
    
    zh_name, latin_name = name.split(' ', 1)
    name = zh_name
    augment_names = [name]
    
    # test original name and possible name, may cause example repeats
    for l, i in _label_index:
        if name.endswith(l) and i >= item2idx['总科 Superfamily']:
            augment_names.append(name[:-len(l)])

    return augment_names


def multi_name_multi_walker(name: str, walkers: Iterable, name_aug=False) -> [Iterable, str, Iterable]:
    
    desc = ''
    web_page = []
    img_urls = []
    global logfile
    
    names = name_augment(name) if name_aug else [name]
    
    for name in names:
        for walker in walkers:
            _webpage, _desc, _img_urls = walker(name)
            desc = desc + '|' + _desc
            web_page.append(_webpage)
            img_urls.extend(_img_urls)
            # text_img_pairs[walker.__name__][name] = (_webpage, _desc, _img_urls)
            
    if len(walkers) > 1 or len(names) > 1:
        logfile.write(f'item {name} find {len(img_urls):2d} imgs totally\n')
    logfile.flush()
    
    return web_page, desc, img_urls


def parse_index_walking(index, output_filepath, walkers):
    
    pbar = tqdm(total=1095, ncols=66)
    
    line = [''] * len(item2idx)
    
    def get_label_index(label: str):
        for l, i in _label_index:
            if l in label:
                return i
        print(label)
        assert 0
    
    def parse_species(data: str, line: list, csvfile):
        
        web_page, desc, img_urls = multi_name_multi_walker(data, walkers)
        
        line[item2idx['特征描述 Description']] = ''
        line[item2idx['百科页面 WebPages']] = f"{','.join(web_page)}"
        line[item2idx['示例图像 Imgs']] = str(img_urls)
        line[item2idx['种 Species']] = data
        
        csvfile.write(','.join(line))
        csvfile.write('\n')
        pbar.update(1)
        
        line[item2idx['特征描述 Description']] = ''
        line[item2idx['百科页面 WebPages']] = ''
        line[item2idx['示例图像 Imgs']] = ''
        line[item2idx['种 Species']] = ''

    def parse_non_species(data, line, csvfile):
        if isinstance(data, list):
            for item in data:
                parse_item(item, line, csvfile)
        else:
            for label, item in data.items():
                
                if label.startswith('种组'):            # ignore '种组'
                    parse_item(item, line, csvfile)
                    continue
                
                label_level = get_label_index(label)
                line[label_level] = label
                
                if label_level >= item2idx['总科 Superfamily']:
                    
                    # walks_in_Baidu(zh_name)
        
                    # print(f'{zh_name}${latin_name}')
                    # walks_in_Baidu(zh_name)
                    # walks_in_Wiki(zh_name)
        
                    web_page, desc, img_urls = multi_name_multi_walker(label, walkers)
                    
                    line[item2idx['特征描述 Description']] = ''
                    line[item2idx['百科页面 WebPages']] = f"{','.join(web_page)}"
                    line[item2idx['示例图像 Imgs']] = str(img_urls)
                    
                    # record 
                    csvfile.write(','.join(line))
                    csvfile.write('\n')
                    pbar.update(1)
                    
                    line[item2idx['特征描述 Description']] = ''
                    line[item2idx['百科页面 WebPages']] = ''
                    line[item2idx['示例图像 Imgs']] = ''
                
                parse_item(item, line, csvfile)
                line[label_level] = ''

    def parse_item(data, line, csvfile):
        if isinstance(data, str):
            parse_species(data, line, csvfile)
        else:
            parse_non_species(data, line, csvfile)

    # parse index
    with open(output_filepath, 'w', encoding='utf8') as f:
        f.write(csv_index + '\n')
        for item in index:
            parse_item(item, line, f)


def parse_index(index, output_filepath):
    
    line = [''] * len(item2idx)
    
    def get_label_index(label: str):
        for l, i in _label_index:
            if l in label:
                return i
        print(label)
        assert 0

    def parse_species(data: str, line: list, csvfile):
        # zh_name, latin_name = data.split(' ', 1)
        
        line[item2idx['种 Species']] = data
        
        csvfile.write(','.join(line))
        csvfile.write('\n')
        
        line[item2idx['特征描述 Description']] = ''
        line[item2idx['百科页面 WebPages']] = ''
        line[item2idx['示例图像 Imgs']] = ''
        line[item2idx['种 Species']] = ''

    def parse_non_species(data, line, csvfile):
        if isinstance(data, list):
            for item in data:
                parse_item(item, line, csvfile)
        else:
            for label, item in data.items():
                
                if label.startswith('种组'):            # ignore '种组'
                    parse_item(item, line, csvfile)
                    continue
                
                label_level = get_label_index(label)
                line[label_level] = label
                
                if label_level >= item2idx['总科 Superfamily']:
                    
                    # record 
                    csvfile.write(','.join(line))
                    csvfile.write('\n')
                    
                    line[item2idx['特征描述 Description']] = ''
                    line[item2idx['百科页面 WebPages']] = ''
                    line[item2idx['示例图像 Imgs']] = ''
                
                parse_item(item, line, csvfile)
                line[label_level] = ''

    def parse_item(data, line, csvfile):
        if isinstance(data, str):
            parse_species(data, line, csvfile)
        else:
            parse_non_species(data, line, csvfile)            

    # parse index
    with open(output_filepath, 'w', encoding='utf8') as f:
        f.write(csv_index + '\n')
        for item in index:
            parse_item(item, line, f)


def to_csv(index_filepath, output_filepath):
    index = None
    with open(index_filepath, 'r', encoding='utf8') as f:
        try:
            index = yaml.safe_load(f)
            # print(index)
        except yaml.YAMLError as exc:
            print(exc)
            os.exit(-1)
    parse_index(index, output_filepath)  # 1077 lines csv, 1119 - 24 目 - 19 种组 + 1 csv_head


def fill_imgs_blank(line, walkers):
    
    for i in range(item2idx['种 Species'], -1, -1):
        if line[i]:
            name = line[i]
            break
    
    # if name == '白眉长臂猿属 Hoolock':
    #     print('debug')
    
    img_list = eval(line[item2idx['示例图像 Imgs']]) if line[item2idx['示例图像 Imgs']] else []
    webpage_list = set() if line[item2idx['百科页面 WebPages']] == '' else set(line[item2idx['百科页面 WebPages']].split('|'))
    
    if any('.png' in url for alt, url in img_list):
        img_list = []
        webpage_list = set()
    
    if len(img_list) == 0 and len(webpage_list) == 0:
        web_page, desc, img_urls = multi_name_multi_walker(name, walkers)
        img_list.extend(img_urls)
        for page in web_page:
            webpage_list.add(page)
    
    img_list = [[alt, url] for alt, url in img_list if not any(keyword in alt for keyword in ['分布', '演变', '迁移', '画', '字', '过程', '示意'])]
    
    if any('.png' in url for url in img_list):
        print('png EXISTS!')
    
    # line[item2idx['特征描述 Description']] = ''
    line[item2idx['百科页面 WebPages']] = f"\"{','.join(webpage_list)}\""
    line[item2idx['示例图像 Imgs']] = '"' + str(img_list) + '"'


def fill_fields(csv_filepath, output_filepath, fill_func, walkers):
    linecnt = 0
    pbar = tqdm(total=1077)
    with open(output_filepath, 'r+' if os.path.exists(output_filepath) else 'w+', encoding='utf8') as f:
        linecnt = len(f.readlines())
        if linecnt == 0:
            f.write(csv_index + '\n')
        pbar.update(linecnt)
    with open(csv_filepath, 'r', encoding='utf8') as csv_file, open(output_filepath, 'a', encoding='utf8') as f:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            if csv_reader.line_num <= linecnt:
                continue
            # fill fields
            fill_func(line, walkers)
            
            f.write(','.join(line) + '\n')
            f.flush()
            pbar.update(1)


def download_images(imgcsv: str, output_dir: str):
    """根据csv文件中的链接下载图像，并保存最终的csv

    Args:
        imgcsv (pd.DataFrame): imgsv2.csv
        output_dir (str): 数据集输出目录
    """
    
    data = pd.read_csv(imgcsv)
    
    for i in range(len(data)):
        image_text_pairs = data.iloc[i]['示例图像 Imgs']
        


if __name__ == '__main__':
    
    log_filepath = 'datacollect/walker.log.txt'
    index_filepath = 'datacollect/index.yaml'
    tempcsv_filepath = 'datacollect/imgs_with_png.csv'
    output_filepath = 'datacollect/imgsv2.csv'
    
    # global text_img_pairs
    # text_img_pairs = DefaultDict()
    
    # global logfile
    # logfile = open(log_filepath, 'a' if os.path.exists(log_filepath) else 'w', encoding='utf8')
    # initialize_browser()
    
    # to_csv(index_filepath, tempcsv_filepath)
    # fill_fields(tempcsv_filepath, output_filepath, fill_imgs_blank, [walks_in_Baidu_by_driver])
    
    # driver.quit()
    
    # test crawler
    # print(walks_in_Baidu('鼹鼠科'))
    
    
    # download images & output final csv
    download_images
