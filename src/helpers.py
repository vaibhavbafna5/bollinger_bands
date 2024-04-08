"""Starter code to create a database running in local container."""

from pymongo import MongoClient

# connect to Mongo, get DB, and access collection
client = MongoClient("mongodb://localhost:27017/")
mydb = client["database_name"]
mycollection = mydb["mycollection"]

# insert something into the collection to create and populate the collection
mycollection.insert_one({"blah": "foo"})

# print database names to verify DB was created
client.list_database_names()