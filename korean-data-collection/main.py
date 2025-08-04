from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time


def main():
    # Configure Chrome options (optional)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--start-maximized")  # Start browser maximized
    chrome_options.add_argument("--disable-notifications")

    driver = webdriver.Chrome(chrome_options)

    # Navigate to the specific URL
    url = "https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36&tabNo=1"  # Replace with your target URL
    driver.get(url)

    try:
        # Wait for the page to load (implicit wait - waits up to 10 seconds for elements)
        driver.implicitly_wait(10)
        driver.implicitly_wait(10)

        # Find the element you want to click using one of these methods:
        # By ID
        element = driver.find_element(By.ID, "ztree_2_switch")

        # By CSS selector (recommended)
        # element = driver.find_element(By.CSS_SELECTOR, "button.submit-btn")

        # By XPath (if needed for complex selections)
        # element = driver.find_element(By.XPATH, "//div[@class='menu']/a[text()='Login']")

        # Click the element
        element.click()

        print("Successfully clicked the element!")

        time.sleep(10)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Close the browser (use driver.quit() to close all windows)
        driver.quit()


if __name__ == "__main__":
    main()
