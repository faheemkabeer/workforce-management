import sqlite3

# Connect to database
conn = sqlite3.connect('anpr_fines.db')
c = conn.cursor()

# Violations table (one record per vehicle)
c.execute('''
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_number TEXT UNIQUE,
    timestamp DATETIME,
    violation_type TEXT,
    fine_amount INTEGER,
    status TEXT
)
''')

# Vehicles table (optional for owner info)
c.execute('''
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_number TEXT PRIMARY KEY,
    owner_name TEXT,
    owner_contact TEXT
)
''')

# Payments table
c.execute('''
CREATE TABLE IF NOT EXISTS payments (
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id INTEGER,
    payment_date DATETIME,
    amount INTEGER,
    payment_status TEXT,
    FOREIGN KEY (violation_id) REFERENCES violations(id)
)
''')

conn.commit()
conn.close()
print("✅ Database created successfully.")
