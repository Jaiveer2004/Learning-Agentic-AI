from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional

class CustomerProfile(BaseModel):
    customer_id: str
    first_name: str
    last_name: str
    email: str
    join_date: datetime
    kyc_status: str
    country_code: str
    total_transactions: int
    last_active_date: Optional[datetime] = None
    total_spent: float
    
class TransactionRecord(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    timestamp: datetime
    transaction_type: str
    status: str