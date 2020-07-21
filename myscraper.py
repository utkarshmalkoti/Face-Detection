import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import os
import threading

def downloadCeleb(index ,celeb, finished):
    driver = webdriver.Chrome()
    google = driver.get("https://www.google.co.in/imghp?hl=en&tab=wi&ogbl")
    search_bar = driver.find_element_by_css_selector('input[name="q"]')
    search_bar.send_keys(celeb)
    search_bar.send_keys(Keys.ENTER)
    imageTile = 1
    imagesDownloaded = 1
    while(True):
        try:
            img = driver.find_element_by_xpath(f'/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[1]/div[{imageTile}]/a[1]')
            img.click()
            #Delay
            WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.XPATH,'/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img')))
            larger_img_path = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img')

            small_img_path = driver.find_element_by_xpath(f'/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[1]/div[{imageTile}]/a[1]/div[1]/img')
            small_img_src = small_img_path.get_attribute('src')

            j = 0
            while(larger_img_path.get_attribute('src')==small_img_src):
                time.sleep(0.01)
                j += 1
                if j == 400 : 
                    break

            if j == 400:
                continue
            larger_img_src = larger_img_path.get_attribute('src')

            
            if not os.path.exists(f'Celebrities/{celeb}'):
                os.makedirs(f'Celebrities/{celeb}')
        
            img_response = requests.get(larger_img_src)
            if (larger_img_src.split('/')[-1].split('.')[-1]) == "jpg" or (larger_img_src.split('/')[-1].split('.')[-1]) == "png":
                ext = larger_img_src.split('/')[-1].split('.')[-1]
            else:
                ext = 'jpg'
            open('Celebrities/{}/{}.{}'.format(celeb,imagesDownloaded,ext),'wb').write(img_response.content)

            imageTile+=1
            imagesDownloaded += 1
            if imagesDownloaded == 700:
                break
        except:
            imageTile+=1
            print("error")
            
    finished[celeb] = 'yes'
    driver.close()


finished= []
celebs = []
while(1):
    name = input("Enter Celebrity name: ")
    if(name=='-1'):
        break
    else:
        celebs.append(name)
for index, celeb in enumerate(celebs):
   
    finished.append( "not")

    while(finished.count('not') > 4) :
        pass
    threading._start_new_thread(downloadCeleb, (index, celeb, finished))

while( finished.count('not')): 
    pass
