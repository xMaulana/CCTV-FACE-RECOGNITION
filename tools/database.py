import pymysql.cursors
import yaml
import datetime
from typing import Union, Callable, Tuple, Optional, Annotated
import sqlite3
import time
import pymysql
import traceback

class Database:
    def __init__(self, config:dict, delay:float=.1, retry:int=10):
        self.config = config
        self.delay=delay
        self.retry=retry
        try:
            self.connection = pymysql.connect(
                host=self.config["host"],
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
                port=3306,
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=self.delay
            )
            print("Database Connected")
        except pymysql.MySQLError as e:
            print(traceback.format_exc())
            raise ConnectionError("Not Connected to Database Error")
        
        self.cursor = self.connection.cursor()
        
        self.insert_query = """
            INSERT INTO {0} (employee_id, name, location, camera_name, status, remark, created_at)
            VALUES ('{{0}}','{{1}}','{{2}}','{{3}}','{{4}}','{{5}}','{{6}}')
            """.format(self.config['table'])
        
    def check_connection(self):
        try:
            self.connection.ping(reconnect=True)
            return True
        except pymysql.MySQLError:
            return False
    
    def insert(self, data:list|set):
        assert len(data) == 7, "Need 7 data"
        try:
            self.cursor.execute(self.insert_query.format(*data))
            self.connection.commit()
            return True
        except pymysql.MySQLError as e:
            print(f"kesalahan menyisipkan data \n{traceback.format_exc()}")
            return False

    def close(self):
        if self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            return True
        return False

    def show_all(self, command:str=None, table_index:int=0):
        for i in range(self.retry):
            try:
                if command is not None:
                    self.cursor.execute(command)
                else:
                    self.cursor.execute(f"SELECT * FROM {self.config[self.sqlconf]['table'][table_index]}")
                ret = self.cursor.fetchall()

                return ret
            except sqlite3.OperationalError as e:
                print(f"ERROR READING, retrying ({i}) | {e}")
                time.sleep(0.1)
        return []
