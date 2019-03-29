import json
import graphql
from graphql.language import ast


def serialize_json(value):
    # print(f'ser_js {type(value)} {value}')
    s = json.dumps(value)
    # print(f'* ser post {type(s)} {s}')
    return s


def parse_value_json(value):
    # print(f'pv_js {type(value)} {value}')
    d = json.loads(value)
    # print(f'* pv post {type(d)} {d}')
    return d


def parse_literal_json(node):
    # print(f'pl_js {node}')
    if isinstance(node, ast.StringValue):
        d = json.loads(node.value)
        # print(f'* lit post {type(d)} {d}')
        return d


scalars = {
    'JSON': {
        'description': "JSON GraphQL scalar",
        'parse_literal': parse_literal_json,
        'parse_value': parse_value_json,
        'serialize': serialize_json
    }
}
