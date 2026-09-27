from pydantic import BaseModel ,EmailStr
from typing import Optional

class MyModel(BaseModel):
    name: str = 'Aman'
    age: Optional[int] = None
    email : EmailStr

new_student = {'age': 23, 'name': 'Sohail', 'email': 'aman.com'}

student = MyModel(**new_student)

print(student)           