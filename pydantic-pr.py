from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    is_active: bool

input_data = {
    'id': '101a',
    'name': 'Rahul',
    'is_active': True
}

try:
    user = User(**input_data) # unpack

    print(user)
except ValueError as e:
    print('valuerror!')