import logging
import os
import pymongo
import pandas as pd
import smtplib, ssl
import sys

from dotenv import load_dotenv
from email.message import EmailMessage
from typing import Optional

load_dotenv()

MONGO_URL = os.environ.get("MONGO_URL")
EMAIL_PW = os.environ.get("EMAIL_PASSWORD")

# configure basic settings for the logging system
log_format = '[%(asctime)s] [%(levelname)s] [%(message)s] [%(filename)s:%(lineno)d]'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)  # set logging to output to standard output
    ]
)


class MongoInstance:
    """Wrapper class used for interacting with Mongo instance."""

    def __init__(self):
        """Attach to Mongo instance via URL, bomb out if it doesn't work."""
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_URL)
        except Exception as e:
            logging.error("Failed to create MongoClient", e)
            raise RuntimeError("Couldn't create a MongoClient %s", e)
        
    def write_dataframe_to_mongo(self, ticker: str, bollinger_bands_df: pd.DataFrame):
        """Upload DataFrame for a given ticker to Mongo."""

        db = self.mongo_client["bollinger_bands"]
        if ticker not in db.list_collection_names():
            logging.warning(f"{ticker} is not a collection in DB")
        
        collection = db[ticker]
        bollinger_bands_df.reset_index(inplace=True)
        values = list(bollinger_bands_df.T.to_dict().values())
        collection.insert_many(values)

    def write_last_row_to_mongo(self, ticker: str, bollinger_bands_df: pd.DataFrame):
        """Uploads entire DataFrame if it isn't in Mongo, otherwise appends last row."""

        db = self.mongo_client["bollinger_bands"]
        if ticker not in db.list_collection_names():
            self.write_dataframe_to_mongo(ticker, bollinger_bands_df)
            return
        
        collection = db[ticker]
        bollinger_bands_df.reset_index(inplace=True)
        last_row = bollinger_bands_df.iloc[-1].to_dict()
        collection.insert_one(last_row)

    def read_dataframe_from_mongo(self, ticker: str) -> Optional[pd.DataFrame]:
        """Read ticker data from Mongo into DataFrame."""

        db = self.mongo_client["bollinger_bands"]
        if ticker not in db.list_collection_names():
            logging.error(f"{ticker} is not a collection in DB")
            return None
        
        collection = db[ticker]
        cursor = collection.find()
        list_cur = list(cursor)

        df = pd.DataFrame(list_cur)
        return df

    def read_last_row_from_mongo(self, ticker: str) -> Optional[pd.Series]:
        """Read the last row for a given ticker from MongoDB."""

        db = self.mongo_client["bollinger_bands"]
        if ticker not in db.list_collection_names():
            logging.error(f"{ticker} is not a collection in DB")
            return None
        
        collection = db[ticker]
        cursor = collection.find().sort('_id', pymongo.DESCENDING).limit(1)
        docs = [doc for doc in cursor]
        if not docs:
            return None

        last_row = pd.Series(docs[0])
        return last_row

 
class EmailSender:
    """Wrapper class to blast emails from sender to receiver with a subject."""

    def __init__(self, port: int, smtp_server: str):
        """Initialize the object with the port to use & smtp server."""
        self.port: int = port
        self.smtp_server: str = smtp_server

    def send_email(self, sender_email: str, receiver_email: str, subject: str, body: str="") -> bool:
        """Send email from desired sender to desired receiver with subject and optional body."""
        msg = EmailMessage()
        msg.set_content(body)

        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email

        context = ssl.create_default_context()
        try:
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(sender_email, EMAIL_PW)
                server.send_message(msg)
        except Exception as e:
            logging.error(f"Failed to send email with error: {e}")
            return False

        return True
            
    
mongo_instance = MongoInstance()
email_sender = EmailSender(port=587, smtp_server="smtp.gmail.com")