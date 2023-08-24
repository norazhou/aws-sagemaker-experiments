#!/usr/bin/python

import psycopg2
import json


def insert_vector(data):
    sql = """INSERT INTO v3_items (page_number, content, pdf_file_path, embedding) 
             VALUES (%s, %s, %s, %s);"""

    conn = None
    vendor_id = None
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(host=DB_HOST, 
                                database=DB_NAME,
                                user=USER,
                                password=PASSWORD)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        for page_index, page in enumerate(data):
            record_to_insert = (data[page_index]["page_number"], data[page_index]["content"], data[page_index]["pdf_file_path"], data[page_index]["embedding"])
            cur.execute(sql, record_to_insert)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

# Define a name for the output file
pdf_json_file_name = 'embedding.json'

with open(pdf_json_file_name, 'r') as f:
    data = json.load(f)

insert_vector(data)

