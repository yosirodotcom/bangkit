from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pyautogui
import pandas as pd
import json
import re

#################################
# Parameters
sektor_ekonomi = "Informasi Dan Komunikasi"
# sektor_ekonomi = "Perdagangan Besar Dan Eceran Reparasi Dan Perawatan Mobil Dan Sepeda Motor"
# sektor_ekonomi = "Pertanian, Kehutanan, Dan Perikanan"
# sektor_ekonomi = "Kesenian, Hiburan Dan Rekreasi"
# sektor_ekonomi = "Pertambangan Dan Penggalian"
# sektor_ekonomi = "Administrasi Pemerintahan, Pertahanan Dan Jaminan Sosial Wajib"
# sektor_ekonomi = "Jasa Kesehatan Dan Kegiatan Sosial"
# sektor_ekonomi = "Transportasi Dan Pergudangan"
# sektor_ekonomi = "Industri Pengolahan"
# sektor_ekonomi = "Konstruksi"
# sektor_ekonomi = "Pengadaan Air, Pengelolaan Sampah Dan Daur Ulang, Pembuangan Dan Pembersihan Limbah Dan Sampah"
# sektor_ekonomi = "Real Estate"
# sektor_ekonomi = "Jasa Profesional, Ilmiah Dan Teknis"
# sektor_ekonomi = "Penyediaan Akomodasi Dan Penyediaan Makan Minum"
# sektor_ekonomi = "Jasa Pendidikan"
# sektor_ekonomi = "Kegiatan Jasa Lainnya"
# sektor_ekonomi = "Jasa Keuangan Dan Asuransi"
# sektor_ekonomi = "Jasa Persewaan Dan Sewa Guna Usaha Tanpa Hak Opsi, Ketenagakerjaan, Agen Perjalanan Dan Penunjang Usaha Lainnya"
# sektor_ekonomi = "Jasa Perorangan Yg Melayani RT, Keg Yg Menghasilkan Brg & Jasa Oleh RT Yg Digunakan Sendiri Utk Memenuhi Kebutuhan"
# sektor_ekonomi = "Pengadaan Listrik, Gas, Uap Air Panas Dan Udara Dingin"

link_ke = 1
# link_ke = 2
# link_ke = 3
# link_ke = 4
# link_ke = 5
# link_ke = 6
# link_ke = 7
# link_ke = 8
# link_ke = 9
# link_ke = 10
# link_ke = 11
# link_ke = 12
# link_ke = 13
# link_ke = 14
# link_ke = 15
# link_ke = 16
# link_ke = 17
# link_ke = 18
# link_ke = 19
# link_ke = 20

xs, ys = 283, 646
# xs, ys = 320, 686
# xs, ys = 238, 729
# xs, ys = 262, 767
# xs, ys = 293, 814
# xs, ys = 309, 852
# xs, ys = 299, 891
# xs, ys = 260, 928
# xs, ys = 265, 615
# xs, ys = 181, 656
# xs, ys = 260, 693
# xs, ys = 196, 756
# xs, ys = 285, 798
# xs, ys = 321, 839
# xs, ys = 236, 881
# xs, ys = 258, 514
# xs, ys = 254, 555
# xs, ys = 246, 601
# xs, ys = 289, 663
# xs, ys = 304, 721

num_pages_loop_1 = 5
num_pages_loop_2 = 10

xb, yb = 675, 717# normal navigation next
xb2, yb2 = 700, 715

# xb, yb = 594, 714# jika navigasinya cuma 2

# xb, yb = 636, 709# jika navigasinya cuma 4
# xb2, yb2 = 681, 795

#################################

# Set up the Chrome WebDriver with custom user agent
chrome_options = Options()
chrome_options.add_argument(
    "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
url = 'https://www.bi.go.id/id/umkm/database/umkm-layak-dibiayai.aspx'
driver.get(url)
pyautogui.moveTo(283, 646, duration=1)
pyautogui.scroll(-7)
# pyautogui.scroll(-10)
# pyautogui.scroll(-20)


# delay = 1.0
# while True:
#     x, y = pyautogui.position()
#     print(f'Mouse position: x={x}, y={y}')
#
#     # Add a delay
#     time.sleep(delay)


pyautogui.moveTo(xs, ys, duration=1)
time.sleep(2)
pyautogui.click()

def extract_sub_table_data(sub_table_html):
    # Define regular expressions to extract data
    pattern = re.compile(r'<td[^>]*>(.*?)<\/td>', re.DOTALL)

    # Find all matches in the sub_table_html
    matches = re.findall(pattern, sub_table_html)

    # Extract data from the matches
    sub_table_data = {}
    for i in range(0, len(matches), 2):
        key = matches[i].strip()
        value = matches[i + 1].strip()
        sub_table_data[key] = value

    return sub_table_data


def createTable(i):
    try:
        # element = WebDriverWait(driver, 2).until(
        #     EC.presence_of_element_located((By.ID, "carousel-menu-minisite"))
        # )


        driver.execute_script("window.stop();")
        print(f"Scrap page {i+1}.....")

        # Get the HTML source before clicking the link
        html_source_before_click = driver.page_source

        # Find the main table
        soup = BeautifulSoup(html_source_before_click, 'html.parser')
        main_table = soup.find('table', class_='table-striped')

        # Get the column headers
        headers = [th.text.strip() for th in main_table.find('tr', {'class': 'table-header'}).find_all('th')]

        # Get the rows
        rows = []
        for tr in main_table.find_all('tr')[1:]:
            # Extract data from the main table
            row_data = {header: td.text.strip() for header, td in zip(headers, tr.find_all('td'))}

            # Find the link in the row
            link = tr.find('a', class_='umkm-popup-link')

            if link:
                # Extract data from the sub-table using the provided HTML source
                sub_table_html = link.get('data-subtablehtml', '')

                # Extract sub-table data using regular expressions
                sub_table_data = extract_sub_table_data(sub_table_html)

                # Update row_data with sub_table_data
                row_data.update(sub_table_data)

                # Extract data from the link attributes
                link_data = {f"data-{key.lower()}": ' '.join(value).strip() if isinstance(value, list) else value.strip() for key, value in dict(link.attrs).items()}
                row_data.update(link_data)

            # Append the row_data to the rows list
            rows.append(row_data)

        return rows

    except TimeoutException:
        print("Timed out waiting for element to be present")
        return None


# Create a list to store all rows
all_rows = []

for i in range(num_pages_loop_1):
    rows = createTable(i=i)

    if rows:
        all_rows.extend(rows)
        df = pd.DataFrame(all_rows)
        df["sektor_ekonomi"] = sektor_ekonomi
        df.to_excel(f"df{link_ke}.xlsx")

        # Convert the list of dictionaries to a JSON string
        # json_data = json.dumps(rows, indent=2, ensure_ascii=False)
        #
        # # Save the JSON data to a file
        # with open(f"output_{i + 1}.json", "w", encoding="utf-8") as json_file:
        #     json_file.write(json_data)

        pyautogui.scroll(-100)

        # delay = 1.0
        # while True:
        #     x, y = pyautogui.position()
        #     print(f'Mouse position: x={x}, y={y}')
        #
        #     # Add a delay
        #     time.sleep(delay)

        pyautogui.moveTo(xb, yb, duration=1)

        pyautogui.click()

if num_pages_loop_1 >= 5:
    for i in range(5,num_pages_loop_2):
        rows = createTable(i=i)

        if rows:
            all_rows.extend(rows)
            df = pd.DataFrame(all_rows)
            df["sektor_ekonomi"] = sektor_ekonomi
            df.to_excel(f"df{link_ke}.xlsx")

            # # Convert the list of dictionaries to a JSON string
            # json_data = json.dumps(rows, indent=2, ensure_ascii=False)
            #
            # # Save the JSON data to a file
            # with open(f"output_{i + 5}.json", "w", encoding="utf-8") as json_file:
            #     json_file.write(json_data)

            # delay = 1.0
            # while True:
            #     x, y = pyautogui.position()
            #     print(f'Mouse position: x={x}, y={y}')
            #
            #     # Add a delay
            #     time.sleep(delay)

            pyautogui.scroll(-100)
            pyautogui.moveTo(xb2, yb2, duration=1)
            pyautogui.click()

# Convert the list of dictionaries for all rows to a JSON string
# all_json_data = json.dumps(all_rows, indent=2, ensure_ascii=False)

# # Save the JSON data for all rows to a file
# with open("all_output_2.json", "w", encoding="utf-8") as all_json_file:
#     all_json_file.write(all_json_data)

# Optionally, create a DataFrame from all_rows
df = pd.DataFrame(all_rows)
# df["sektor_ekonomi"] = "Perdagangan Besar Dan Eceran Reparasi Dan Perawatan Mobil Dan Sepeda Motor"
# df["sektor_ekonomi"] = "Pertanian, Kehutanan, Dan Perikanan"
df["sektor_ekonomi"] = sektor_ekonomi
df.to_excel(f"df{link_ke}.xlsx")
print(df.info())

