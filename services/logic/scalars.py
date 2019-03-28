import json
import graphql
from graphql.language import ast


def serialize_json(value):
    # print(f'ser_js {value}')
    return json.dumps(value)


def parse_value_json(value):
    # print(f'pv_js {value}')
    return json.loads(value)


def parse_literal_json(node):
    # print(f'pl_js {node}')
    if isinstance(node, ast.StringValue):
        return json.loads(node.value)


scalars = {
    'JSON': {
        'description': "JSON GraphQL scalar",
        'parse_literal': parse_literal_json,
        'parse_value': parse_value_json,
        'serialize': serialize_json
    }
}
