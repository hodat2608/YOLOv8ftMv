import mysql.connector
from tkinter import messagebox, simpledialog


class MySQL_Connection:

    def __init__(self, host, user, passwd, database):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.database = database

    def Connect_MySQLServer_root(self):
        db_connection = mysql.connector.connect(
            host=self.host, user=self.user, passwd=self.passwd, database=self.database
        )
        cursor = db_connection.cursor()
        return cursor, db_connection

    def check_connection(self):
        _, db_connection = self.Connect_MySQLServer()
        try:
            db_connection.ping(reconnect=True, attempts=3, delay=5)
            return True
        except mysql.connector.Error as err:
            messagebox.showinfo(
                "Notification", f"Error connecting to the database: {str(err)}"
            )
            return False

    def reconnect(self):
        _, db_connection = self.Connect_MySQLServer()
        try:
            db_connection.reconnect(attempts=3, delay=5)
            cursor = db_connection.cursor()
            return True
        except mysql.connector.Error as err:
            messagebox.showinfo(
                "Notification", f"Failed to reconnect to the database: {str(err)}"
            )
            return False

    @staticmethod
    def Connect_to_MySQLServer(host, user, passwd, database):
        db_connection = mysql.connector.connect(
            host=host, user=user, passwd=passwd, database=database
        )
        cursor = db_connection.cursor()
        return cursor, db_connection

    def Connect_MySQLServer(self):
        try:
            db_connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.passwd,
            )
            if db_connection.is_connected():
                cursor = db_connection.cursor(dictionary=True)
                return cursor, db_connection
        except Exception as e:
            print(f"Error: {e}")
            return None, None
