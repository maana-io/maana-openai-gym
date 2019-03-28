# Standard
from aiohttp import web
import aiohttp_cors
from functools import lru_cache
import json
import logging
import random
import asyncio
import sys
from jinja2 import Environment
import numpy as np

# GraphQL
from shared.graphiql import GraphIQL
import graphql as gql
from graphql_tools import build_executable_schema

# OpenAI Gym
import gym

# Maana
from CKGClient import CKGClient
from shared.maana_amqp_pubsub import amqp_pubsub, configuration
from settings import SERVICE_ID, SERVICE_PORT, RABBITMQ_ADDR, RABBITMQ_PORT, SERVICE_ADDRESS, REMOTE_KSVC_ENDPOINT_URL

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

ENV_NAME = "FrozenLake-v0"

source_schema = """
    schema {
        query: RootQuery
    }

    type RootQuery {
        learn(env: String!, endpoint: String!, decideFn: String!, observeFn: String!, resetFn: String!, context: String!, episodes: Int!, steps: Int!): Boolean!
        decide(context: String!, state: String!): String!
        observe(context: String!, state: String!, state2: String!, reward: Float!, done: Boolean!, info: String!): String!
        reset(context: String!, actionSpace: SpaceAsInput! observationSpace: SpaceAsInput!): String!
        sampleActionSpace(env: String!): String!
    }

    input SpaceAsInput {
        id: ID!
        shape: [Int!]!
        dtype: String!
    }
"""


resolvers = {
    'RootQuery': {
        'learn': lambda value, info, **args: learn(args['env'], args['endpoint'], args['decideFn'], args['observeFn'], args['resetFn'], args['context'], args['episodes'], args['steps']),
        'decide': lambda value, info, **args: decide(args['context'], args['state']),
        'observe': lambda value, info, **args: observe(args['context'], args['state'], args['state2'], args['reward'], args['done'], args['info']),
        'reset': lambda value, info, **args: reset(args['context'], args['actionSpace'], args['observationSpace']),
        'sampleActionSpace': lambda value, info, **args: sample_action_space(args['env'])
    }
}

executable_schema = build_executable_schema(source_schema, resolvers)

#
# Resolver implementation
#


async def learn(env_name, endpoint, decideFn, observeFn, resetFn, context, episodes, steps):
    print(f'learn(env={env_name},endpoint={endpoint},decideFn={decideFn},observeFn={observeFn},resetFn={resetFn},context={context},episodes={episodes},steps={steps})')
    env = gym.make(env_name)
    context = await call_reset(endpoint, resetFn, context, env)
    for i in range(episodes):
        # print(f'{i}/{episodes}')
        state = env.reset()
        for j in range(steps):
            action = await call_decide(endpoint, decideFn, context, state)
            state2, reward, done, info = env.step(action)

            context = await call_observe(endpoint, observeFn,
                                         context, action, state, state2, reward, done, info)

            if done:
                # env.render()
                break

            state = state2

    print(f'final context={context}')
    return True


def sample_action_space(env_name):
    env = gym.make(env_name)
    return env.action_space.sample()

#
# Testing
#


def try_parse_json(input):
    if isinstance(input, str):
        return json.loads(input)
    return input


async def decide(contextJs, stateJs):
    context = try_parse_json(contextJs)
    state = try_parse_json(stateJs)
    # print(f'decide({context},{state})')

    epsilon = context['epsilon']
    Q = context['Q']

    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = sample_action_space(ENV_NAME)
    else:
        action = np.argmax(Q[state, :])
    return action


async def observe(contextJs, action, stateJs, state2Js, reward, done, info):
    context = try_parse_json(contextJs)
    state = try_parse_json(stateJs)
    state2 = try_parse_json(state2Js)
    # print(
    #     f'observe(action = {action}, context = {context}, state = {state}, state2 = {state2}, reward = {reward}, done = {done}, info = {info})')

    Q = context['Q']
    gamma = context['gamma']
    lr_rate = context['lr_rate']

    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

    # print(f'Q[s, a]={Q[state, action]}, predict={predict}, target={target}, reward={reward}, action={action}, state={state}, state2={state2}, lr_rate={lr_rate}, gamma={gamma}')

    context['Q'] = Q
    return context


async def reset(contextJs, actionSpaceJs, observationSpaceJs):
    context = try_parse_json(contextJs)
    action_space = try_parse_json(actionSpaceJs)
    observation_space = try_parse_json(observationSpaceJs)
    # print(
    #     f'reset(ctx = {context}, actionSpace = {action_space}, observationSpace = {observation_space})')

    context['Q'] = np.zeros((observation_space['n'], action_space['n']))

    return context

#
# Helpers
#


def serialize_space(space):
    obj = dict()
    if isinstance(space, gym.spaces.Box):
        obj['low'] = space.low.tolist()
        obj['high'] = space.high.tolist()
        obj['shape'] = space.shape
        obj['dtype'] = space.dtype.name
    elif isinstance(space, gym.spaces.Discrete):
        obj['n'] = space.n
        obj['shape'] = space.shape
        obj['dtype'] = space.dtype.name
    return json.dumps(obj)


async def call_reset(endpoint, resetFn, context, env):
    return await reset(context, serialize_space(env.action_space), serialize_space(env.observation_space))
    # client = CKGClient(endpoint)
    # return context


async def call_decide(endpoint, decideFn, context, state):
    return await decide(context, state)
    # action = "swim"
    # return action


async def call_observe(endpoint, observeFn, context, action, state, state2, reward, done, info):
    return await observe(context, action, state, state2, reward, done, info)
    # return context


#
# Server
#


async def handle_event(x):
    data_in = x.decode('utf8')
    logger.info("Got event: " + data_in)
    # await handle(data_in)
    return None


def init(loopy):
    asyncio.set_event_loop(loopy)
    app = web.Application(loop=loopy)

    async def graphql(request):
        back = await request.json()
        result = await gql.graphql(executable_schema, back.get('query', ''), variable_values=back.get('variables', ''),
                                   operation_name=back.get(
            'operationName', ''),
            return_promise=True, allow_subscriptions=True)
        data = dict()
        if result.errors:
            data['errors'] = [str(err) for err in result.errors]
        if result.data:
            data['data'] = result.data
        if result.invalid:
            data['invalid'] = result.invalid
        return web.Response(text=json.dumps(data), headers={'Content-Type': 'application/json'})

    # For /graphql
    app.router.add_post('/graphql', graphql, name='graphql')
    app.router.add_get('/graphql', graphql, name='graphql')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    for route in list(app.router.routes()):
        cors.add(route)

    # For graphIQL
    j_env = Environment(enable_async=True)
    gql_view = GraphIQL.GraphIQL(
        schema=executable_schema, jinja_env=j_env, graphiql=True)
    app.router.add_route('*', handler=gql_view,
                         path="/graphiql", name='graphiql')

    loopy.run_until_complete(
        asyncio.gather(
            asyncio.ensure_future(
                loopy.create_server(app.make_handler(),
                                    SERVICE_ADDRESS, SERVICE_PORT)
            ),
            asyncio.ensure_future(
                amqp_pubsub.AmqpPubSub(configuration.AmqpConnectionConfig(RABBITMQ_ADDR, RABBITMQ_PORT, SERVICE_ID)).
                subscribe("fileAdded", lambda x: handle_event(x))
            )
        )
    )

    try:
        logging.info("Started server on {}:{}".format(
            SERVICE_ADDRESS, SERVICE_PORT))
        loopy.run_forever()
    except Exception as e:
        loopy.close()
        logger.error(e)
        sys.exit(-1)
    return None


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        init(loop)
    except KeyboardInterrupt:
        loop.close()
        sys.exit(1)
