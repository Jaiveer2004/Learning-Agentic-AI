import sqlite3

from fastapi import FastAPI, Depends, HTTPException, Query

from database import get_db
from schemas import CustomerProfile, TransactionRecord

app = FastAPI()

@app.get("/customers/{customer_id}", response_model=CustomerProfile)
def get_customers(
    customer_id: str,
    conn: sqlite3.Connection = Depends(get_db)
):
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM customers WHERE customer_id = ?",
        (customer_id,)
    )
    
    customer = cursor.fetchone()
    
    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found"
        )
        
    return dict(customer)



@app.get("/transactions", response_model=list[TransactionRecord])
def get_transactions(
    status: str | None = None,
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    conn: sqlite3.Connection = Depends(get_db)
):
    cursor = conn.cursor()

    # 1. Start with the base SQL query
    query = "SELECT * FROM transactions WHERE 1=1"

    # 2. Store SQL parameter values
    params = []

    # 3. Add status filter if provided
    if status is not None:
        query += " AND status = ?"
        params.append(status)

    # 4. Add pagination
    query += " LIMIT ? OFFSET ?"

    params.extend([limit, offset])

    # 5. Execute the parameterized query
    cursor.execute(query, tuple(params))

    transactions = cursor.fetchall()

    # 6. Convert SQLite rows into dictionaries
    return [dict(transaction) for transaction in transactions]
