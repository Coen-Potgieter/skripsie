import numpy as np
import requests as re


def main():

    data_link = "https://www.dws.gov.za/hydrology/Verified/HyData.aspx?Station=G1H019100.00&DataType=Daily&StartDT=1968-04-23&EndDT=2020-05-22&SiteType=RIV"
    re.get(data_link)
    pass


if __name__ == "__main__":
    main()
