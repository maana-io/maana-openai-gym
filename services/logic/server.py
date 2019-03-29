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
import graphql as gql
from shared.graphiql import GraphIQL
from graphql_tools import build_executable_schema
from scalars import scalars

# OpenAI Gym
import gym
from gym import envs
import retro

# Maana
from CKGClient import CKGClient
from shared.maana_amqp_pubsub import amqp_pubsub, configuration
from settings import SERVICE_ID, SERVICE_PORT, RABBITMQ_ADDR, RABBITMQ_PORT, SERVICE_ADDRESS, REMOTE_KSVC_ENDPOINT_URL

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

source_schema = """
    scalar JSON

    schema {
        query: RootQuery
    }

    type RootQuery {
        train(env: String!, endpoint: String!, decideFn: String!, learnFn: String!, initFn: String!, context: JSON!, episodes: Int!, steps: Int!): JSON!
        play(env: String!, endpoint: String!, decideFn: String!, context: JSON!): JSON!
        sampleActionSpace(env: String!): Space!
        listEnvs: [String!]
        #
        test_decide(context: JSON!, state: JSON!): JSON!
        test_learn(context: JSON!, action: JSON!, state: JSON!, state2: JSON!, reward: Float!, done: Boolean!, inf: JSON!): JSON!
        test_init(context: JSON!, actionSpace: SpaceAsInput! observationSpace: SpaceAsInput!): JSON!
    }

    enum SpaceType {
        Discrete,
        BoxR,
        BoxZ
    }

    type Space {
        type: SpaceType!
        n: Int
        lowR: [Float!]
        highR: [Float!]
        lowZ: [Int!]
        highZ: [Int]
        shape: [Int!]
        dtype: String
    }

    input SpaceAsInput {
        type: SpaceType!
        n: Int
        lowR: [Float!]
        highR: [Float!]
        lowZ: [Int!]
        highZ: [Int]
        shape: [Int!]!
        dtype: String!
    }
"""


resolvers = {
    'RootQuery': {
        'train': lambda value, info, **args: train(args['env'], args['endpoint'], args['decideFn'], args['learnFn'], args['initFn'], args['context'], args['episodes'], args['steps']),
        'play': lambda value, info, **args: play(args['env'], args['endpoint'], args['decideFn'], args['context']),
        'sampleActionSpace': lambda value, info, **args: sample_action_space(args['env']),
        'listEnvs': lambda value, info, **args: list_envs(),
        #
        'test_decide': lambda value, info, **args: decide(args['context'], args['state']),
        'test_learn': lambda value, info, **args: learn(args['context'], args['action'], args['state'], args['state2'], args['reward'], args['done'], args['inf']),
        'test_init': lambda value, info, **args: init(args['context'], args['actionSpace'], args['observationSpace'])
    }
}


executable_schema = build_executable_schema(source_schema, resolvers, scalars)

#
# Resolver implementation
#

ENV_NAME = ""


async def train(env_name, endpoint, decideFn, learnFn, initFn, context, episodes, steps):
    # print(f'train(env={env_name},endpoint={endpoint},decideFn={decideFn},learnFn={learnFn},initFn={initFn},context={context},episodes={episodes},steps={steps})')
    global ENV_NAME
    ENV_NAME = env_name
    env = try_make_env(env_name)
    client = CKGClient(endpoint, is_auth=False)
    context = await call_init(client, initFn, context, env)
    wins = 0
    tot = 0
    for i in range(episodes):
        # print(f'{i}/{episodes}')
        state = env.reset()
        for j in range(steps):
            tot += 1
            action = await call_decide(client, decideFn, context, state)
            state2, reward, done, inf = env.step(action)

            context = await call_learn(client, learnFn,
                                       context, action, state, state2, reward, done, inf)

            if done:
                env.render()
                win = False
                if reward > 0:
                    wins += 1
                    win = True

                print(f'  steps: {win} {j} {wins}/{tot}')
                break

            state = state2

    # print(f'final context={context}')
    return context


async def play(env_name, endpoint, decideFn, context):
    # print(
    #     f'play(env={env_name},endpoint={endpoint},decideFn={decideFn},context={context})')
    env = try_make_env(env_name)
    client = CKGClient(endpoint, is_auth=False)
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = await call_decide(client, decideFn, context, state)
        state2, _, done, _ = env.step(action)
        env.render()
        state = state2

    return context


def sample_action_space(env_name):
    env = try_make_env(env_name)
    return env.action_space.sample()


def list_envs():
    x = retro.data.list_games() + \
        [env_spec.id for env_spec in envs.registry.all()]
    x.sort()
    return x


#
# Testing
#


async def init(context, action_space, observation_space):
    # print(
    #     f'init(contex = {type(context)}, actionSpace = {action_space}, observationSpace = {observation_space})')

    # if observation_space['type'] == 'Discrete' and action_space['type'] == 'Discrete':
    rows = observation_space['n']
    cols = action_space['n']

    context['Q_rows'] = rows
    context['Q_cols'] = cols
    context['Q'] = np.zeros((rows,  cols)).tolist()

    return context


async def decide(context, state):
    # print(f'decide({type(context)},{type(state)})')

    epsilon = context['epsilon']
    Q = np.array(context['Q'])

    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = sample_action_space(ENV_NAME)
    else:
        action = np.argmax(Q[int(state), :])
    return int(action)


async def learn(context, action, state, state2, reward, done, inf):
    # print(
    #     f'learn(action = {action}, context = {context}, state = {state}, state2 = {state2}, reward = {reward}, done = {done}, inf = {inf})')

    Q = np.array(context['Q'])
    gamma = context['gamma']
    alpha = context['alpha']

    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + alpha * (target - predict)

    context['Q'] = Q.tolist()

    return context


#
# Helpers
#

def try_make_env(env_name):
    try:
        env = gym.make(env_name)
    except:
        env = retro.make(env_name)
    return env


def create_space_dict_json(space):
    obj = dict()
    if isinstance(space, gym.spaces.Box):
        if np.issubdtype(space.low.dtype, np.float32):
            obj['type'] = 'BoxR'
            obj['lowR'] = space.low.tolist()
            obj['highR'] = space.high.tolist()
        elif np.issubdtype(space.low.dtype, np.uint8):
            obj['type'] = 'BoxZ'
            obj['lowZ'] = space.low.tolist()
            obj['highZ'] = space.high.tolist()
        obj['shape'] = space.shape
        obj['dtype'] = space.dtype.name
    elif isinstance(space, gym.spaces.Discrete):
        obj['type'] = 'Discrete'
        obj['n'] = space.n
        obj['shape'] = space.shape
        obj['dtype'] = space.dtype.name
    # print('obj', obj['type'])
    return obj


async def call_init(client, initFn, context, env):
    query = f"""
      query init($context: JSON!, $actionSpace: SpaceAsInput!, $observationSpace: SpaceAsInput!){{
        {initFn}(context: $context, actionSpace: $actionSpace, observationSpace: $observationSpace)
      }}
    """
    variables = {'context': json.dumps(context), 'actionSpace': create_space_dict_json(env.action_space),
                 'observationSpace': create_space_dict_json(env.observation_space)}
    d = await client.async_query(query, variables)
    if 'errors' in d:
        raise Exception(d['errors'][0])
    return json.loads(d['data'][initFn])


async def call_decide(client, decideFn, context, state):
    query = f"""
      query decide($context: JSON!, $state: JSON!){{
        {decideFn}(context: $context, state: $state)
      }}
    """
    variables = {'context': json.dumps(
        context), 'state': json.dumps(int(state))}
    d = await client.async_query(query, variables)
    if 'errors' in d:
        raise Exception(d['errors'][0])
    return json.loads(d['data'][decideFn])


async def call_learn(client, learnFn, context, action, state, state2, reward, done, inf):
    query = f"""
      query learn($context: JSON!, $action: JSON!, $state: JSON!, $state2: JSON!, $reward: Float!, $done: Boolean!, $inf: JSON!){{
        {learnFn}(context: $context, action: $action, state: $state, state2: $state2, reward: $reward, done: $done, inf: $inf)
      }}
    """
    variables = {
        'context': json.dumps(context),
        'action': json.dumps(int(action)),
        'state': json.dumps(int(state)),
        'state2': json.dumps(int(state2)),
        'reward': reward,
        'done': done,
        'inf': json.dumps(inf),
    }
    d = await client.async_query(query, variables)
    if 'errors' in d:
        raise Exception(d['errors'][0])
    return json.loads(d['data'][learnFn])


#
# Server
#


async def handle_event(x):
    data_in = x.decode('utf8')
    logger.info("Got event: " + data_in)
    # await handle(data_in)
    return None


def init_server(loopy):
    asyncio.set_event_loop(loopy)
    app = web.Application(loop=loopy)

    async def graphql(request):
        back = await request.json()
        # print(f'----- RAW {back}')
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
        init_server(loop)
    except KeyboardInterrupt:
        loop.close()
        sys.exit(1)
