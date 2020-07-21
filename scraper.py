# import requests
# from bs4 import BeautifulSoup
# name= ['scarlett+johansson','emilia+clarke']
# for i in name:
#     url='https://www.google.com/search?q='+i+'&source=lnms&tbm=isch&sa=X&ved=2ahUKEwi20ryc9ZXqAhXuyzgGHS2sBrAQ_AUoAXoECCEQAw&biw=1920&bih=969'
#     source_code = requests.get(url)
#     html_code = source_code.text
#     soup = BeautifulSoup(html_code)
#     for link in soup.find_all("img"):
#         src=link.get('src')
#         print(src)



import requests
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()
driver.get("https://www.google.co.in/search?q=scarlett+johansson&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjKk6Cx1ZfqAhWDwTgGHd1VApMQ_AUoAXoECCEQAw&biw=1920&bih=969")

for i in range(1):
    print(i)
    try :
        element = driver.find_element_by_xpath(f"/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[1]/div[{i}]/a[1]")
        element.click()
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img"))
        )

        smallerImgElement = driver.find_element_by_xpath(f"/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[1]/div[{i}]/a[1]/div[1]/img")
        smallerImg = smallerImgElement.get_attribute("src")
        
        image = driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img")
        while (image.get_attribute("src") == smallerImg):
            time.sleep(0.1)


        url = image.get_attribute('src')

        # print(url)
        ext = url.split('/')[-1].split('.')[-1]
        r = requests.get(url)
        open(f"{i}.{ext}", 'wb').write(r.content)
    except:
        pass

driver.close()