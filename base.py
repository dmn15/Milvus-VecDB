from pydantic import BaseModel
from typing import List

class doc_ret(BaseModel):
    test_messages: List[str]
