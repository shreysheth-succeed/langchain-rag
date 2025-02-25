import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()

DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")

def get_db_connection():
    """Establishes a connection to SQL Server using Windows Authentication."""
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_NAME};"
        "Trusted_Connection=yes;"
    )
    return conn