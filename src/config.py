import os, sys
import pymongo
from dataclasses import dataclass

@dataclass
class EnvironmentalVariables:
    mongo_db_url:str=os.getenv("MONGO_DB_URL")

env_var = EnvironmentalVariables()
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)

# if __name__=="__main__":
#     print(mongo_client)