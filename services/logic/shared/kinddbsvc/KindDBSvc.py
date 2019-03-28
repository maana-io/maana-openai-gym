import sys
import json
import string
import logging
import asyncio
import aiohttp
from settings import KINDDB_SERVICE_URL, LOG_LEVEL

kindDetailsFragment = """
id
name
description
serviceId
thumbnailUrl
nameField
isPublic
schema {
  id
  name
  description
  type
  typeKindId
  modifiers
  kind {
    id
    name
  }
  hide
  autoFocus
  displayAs
  readonly
}
"""

InstanceDetailsFragment = """
    id
    kindId
    kind {
      schema {
        id
        name
        type
        modifiers
        typeKindId
      }
    }
    fieldIds
    fieldValues {
      ID
      STRING
      INT
      FLOAT
      BOOLEAN
      DATE
      TIME
      DATETIME
      JSON
      KIND
      l_ID
      l_STRING
      l_INT
      l_FLOAT
      l_BOOLEAN
      l_DATE
      l_TIME
      l_DATETIME
      l_JSON
      l_KIND
    }
"""

InstanceSetDetailsFragment = """
    kindId
    kind {
      schema {
        id
        name
        type
        modifiers
        typeKindId
      }
    }
    fieldIds
    records {
      ID
      STRING
      INT
      FLOAT
      BOOLEAN
      DATE
      TIME
      DATETIME
      JSON
      KIND
      l_ID
      l_STRING
      l_INT
      l_FLOAT
      l_BOOLEAN
      l_DATE
      l_TIME
      l_DATETIME
      l_JSON
      l_KIND
    }
  """

LinkDetailsFragment = """
    id
    relation {
      id
    }
    fromKind {
      id
    }
    toKind {
      id
    }
    fromInstance {
      id
    }
    toInstance {
      id
    }
    name
    weight
    fromOffset
    fromSpan
    toOffset
    toSpan
"""


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL)


class KindDBSvc:

    def _create_fieldValueObject(self, fType, value, modifiers):
        isList = "LIST" in modifiers
        fieldValueObject = None

        if fType == "ID":
            fieldValueObject = ({"l_ID": value} if isList else {"ID": value})
        if fType == "STRING":
            fieldValueObject = ({"l_STRING": value} if isList else {"STRING": value})
        if fType == "INT":
            fieldValueObject = ({"l_INT": value} if isList else {"INT": value})
        if fType == "FLOAT":
            fieldValueObject = ({"l_FLOAT": value} if isList else {"FLOAT": value})
        if fType == "BOOLEAN":
            fieldValueObject = ({"l_BOOLEAN": value} if isList else {"BOOLEAN": value})
        if fType == "DATE":
            fieldValueObject = ({"l_DATE": value} if isList else {"DATE": value})
        if fType == "TIME":
            fieldValueObject = ({"l_TIME": value} if isList else {"TIME": value})
        if fType == "DATETIME":
            fieldValueObject = ({"l_DATETIME": value} if isList else {"DATETIME": value})
        if fType == "BOOLEAN":
            fieldValueObject = ({"l_BOOLEAN": value} if isList else {"BOOLEAN": value})
        if fType == "JSON":
            fieldValueObject = ({"l_JSON": value} if isList else {"JSON": value})
        if fType == "KIND":
            fieldValueObject = ({"l_KIND": value} if isList else {"KIND": value})

        return fieldValueObject

    def _instanceSetFromObjects(self, schema, objects):
        fieldIds = [f['id'] for f in schema if f['name'] != 'id']

        instanceIds = []
        records = []

        for obj in objects:
            record = []
            for field in schema:
                val = obj.get(field['name'], None)

                if field['name'] == 'id':
                    instanceIds.append(val)
                else:
                    record.append(self._create_fieldValueObject(field['type'], val, field['modifiers']))

            records.append(record)

        return fieldIds, instanceIds, records

    def _object_to_addInstanceInput(self, kind, instance):
        addInstanceInput = {
            "kindId": kind['id'],
            "id": instance["id"],
            "fieldIds": [],
            "fieldValues": []
        }

        for k, v in instance.items():
            lis = list(filter(lambda x: x["name"] == k, kind["schema"]))
            if len(lis) > 0:
                field = lis[0]
            else:
                raise IOError("Field name specified in instance input not found in kind schema.")
            addInstanceInput["fieldIds"].append(field["id"])
            addInstanceInput["fieldValues"].append(self._create_fieldValueObject(field["type"], v, field["modifiers"]))

        return addInstanceInput

    def _check_response(self, json_resp):

        if 'errors' in json_resp.keys():
            logger.error(json_resp['errors'])
            raise RuntimeError(json_resp['errors'])
        else:
            pass

    def __init__(self, tenantId, loop=asyncio.get_event_loop(), svcUrl = KINDDB_SERVICE_URL):

        self.loop = loop

        if tenantId is None or len(str(tenantId).strip()) == 0:
            raise ValueError("Missing argument: tenantId")
        else:
            self.tenantId = tenantId

        if svcUrl is None or len(svcUrl.strip()) == 0:
            raise ValueError("Missing argument: svcUrl")
        else:
            self.svcUrl = svcUrl

        self.headers = {"Content-Type": "application/json"}
        try:
            self.session = aiohttp.ClientSession(loop=loop)
        except Exception as e:
            logger.error(e)

    async def close(self):
        await self.session.close()

    async def getKind(self, kindId, kindName):
        query = string.Template(
            """query( $tenantId: ID!, $kindId: ID, $kindName: String) {
                kind(tenantId: $tenantId, id: $kindId, name: $kindName) {
                    $kindFragment
                }
            }
        """
        )
        variables = {
            "tenantId": self.tenantId,
            "kindId": kindId,
            "kindName": kindName
        }
        to_post = {
            "query": query.safe_substitute(kindFragment=kindDetailsFragment),
            "variables": variables
        }
        logger.info("getKind kn: {} kid: {}".format(kindName, kindId))
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()
        if out["data"]["kind"] is None:
            logger.error("No data received from kindDB")
            raise RuntimeError("No data received from kindDB")
        self._check_response(out)
        return out["data"]

    async def getKindID(self, kindName):
        res = await self.getAllInstances(kindName="Kind")
        base = res.get("allInstances")
        if base is not None:
            records = base.get("records")
            matches = [r for r in records if r[1].get("STRING") == kindName]
            if len(matches) > 0:
                return matches[0][0].get("ID")

        return None

    async def allKinds(self):
        query = string.Template("""
            query($tenantId: ID!) {
                allKinds(tenantId: $tenantId) {
                    $kindFragment
                }
            }
        """)
        variables = {
            "tenantId": self.tenantId
        }
        to_post = {
            "query": query.safe_substitute(kindFragment=kindDetailsFragment),
            "variables": variables
        }
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()
        self._check_response(out)
        return out["data"]

    async def getInstance(self, kindId, kindName, instanceId):
        query = string.Template("""
            query($tenantId: ID!, $instanceRef: InstanceRefInput!) {
                instance(tenantId: $tenantId, instanceRef: $instanceRef) {
                  $instanceDetailsFragment
                }
            }
        """)
        variables = {
            "tenantId": self.tenantId,
            "instanceRef": {
                "id": instanceId,
                "kindId": kindId,
                "kindName": kindName
            }
        }
        to_post = {
            "query": query.safe_substitute(instanceDetailsFragment=InstanceDetailsFragment),
            "variables": variables
        }
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()

        logger.info("getInstance kid: {}".format(kindId))
        self._check_response(out)
        return out["data"]

    async def getInstanceByName(self, kindName, instanceId):
        k_id = await self.getKindID(kindName=kindName)
        return await self.getInstance(kindId=k_id, instanceId=instanceId, kindName=kindName)

    async def addRelation(self, addRelationInput):
        query = string.Template(
            """mutation($tenantId: ID!, $addRelationInput: AddRelationInput!) {
                addRelation(tenantId: $tenantId, input: $addRelationInput)
            }
        """
        )
        variables = {
            "tenantId": self.tenantId,
            "addRelationInput": addRelationInput
        }
        to_post = {
            "query": query.safe_substitute(kindFragment=kindDetailsFragment),
            "variables": variables
        }
        logger.info("Add relation: {} ".format(addRelationInput))
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()
        self._check_response(out)
        return out["data"]

    async def getInstanceByName(self, kindName, instanceId):
        k_id = await self.getKindID(kindName=kindName)
        return await self.getInstance(kindId=k_id, kindName=kindName, instanceId=instanceId)

    async def getLink(self, linkId):
        query = string.Template("""
            query($tenantId: ID!, $id: ID!) {
                link(tenantId: $tenantId, id: $id) {
                  $linkDetailsFragment
                }
              }
        """)
        variables = {
            "tenantId": self.tenantId,
            "id": linkId
        }
        to_post = {
            "query": query.safe_substitute(linkDetailsFragment=LinkDetailsFragment),
            "variables": variables
        }
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()
        logger.info("getLink id: {}".format(linkId))
        self._check_response(out)
        return out["data"]

    async def addLink(self, addLinkInput):
        query = string.Template("""
            mutation($tenantId: ID!, $addLinkInput: AddLinkInput!) {
                addLink(tenantId: $tenantId, input: $addLinkInput)
            }
        """)
        variables = {
            "tenantId": self.tenantId,
            "addLinkInput": addLinkInput
        }
        to_post = {
            "query": query.template,
            "variables": variables
        }
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()
        logger.info("addLink: {}".format(addLinkInput))
        self._check_response(out)
        return out["data"]

    async def addLinks(self, addLinkInputs):
        query = string.Template("""
            mutation($tenantId: ID!, $addLinkInputs: [AddLinkInput]!) {
                addLinks(tenantId: $tenantId, input: $addLinkInputs)
            }
        """)
        variables = {
            "tenantId": self.tenantId,
            "addLinkInputs": addLinkInputs
        }
        to_post = {
            "query": query.template,
            "variables": variables
        }
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()
        logger.info("addLinks: {}".format(addLinkInputs))
        self._check_response(out)
        return out["data"]

    async def getAllInstances(self, kindId=None, kindName=None, fieldIds=None, take=0, recursion=False):
        query = string.Template("""
              query($tenantId: ID!, $kindId: ID, $kindName: String, $take: Int) {
                allInstances(
                  tenantId: $tenantId
                  kindId: $kindId
                  kindName: $kindName
                  take: $take
                ) {
                  $InstanceSetDetails
                }
              }
        """)
        variables = {
            "tenantId": self.tenantId,
            "kindId": kindId,
            "kindName": kindName,
            "fieldIds": fieldIds,
            "take": take
        }
        to_post = {
            "query": query.safe_substitute(InstanceSetDetails=InstanceSetDetailsFragment),
            "variables": variables
        }
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()

        if out["data"]["allInstances"] is None:
            return None

        if recursion:
            schema = out['data']['allInstances']['kind']['schema']
            schema_map = dict(enumerate(schema))
            ks = {i: f for (i, f) in schema_map.items() if f['type'] == 'KIND'}
            if len(ks) > 0:
                insts = out['data']['allInstances']['records']
                logger.debug("Beginning Nesting")
                nested_data = {i: await self.getAllInstances(kindId=f['typeKindId'], recursion=True)
                               for (i, f) in ks.items()}
                new = []
                for r in insts:
                    for i, d in nested_data.items():
                        if d is not None:
                            to_replace = r[i]
                            if len(ks[i]['modifiers']) > 0:
                                new_to_replace = []
                                for a in to_replace['l_KIND']:
                                    new_to_replace.append(
                                        [rs for rs in d['allInstances']['records'] if rs[0]['ID'] == a][0]
                                    )
                                r[i]['l_KIND'] = new_to_replace
                                new.append(r)
                            elif to_replace['KIND'] is not None:
                                try:
                                    r[i]['KIND'] = \
                                        [r for r in d['allInstances']['records'] if r[0]['ID'] == to_replace['KIND']][0]
                                except Exception:
                                    logger.error(
                                        "Invalid kind id in KIND field for: {} kid: {}".format(kindName, kindId))
                                    logger.error(str(ks[i]))
                                    pass
                            else:
                                r[i]['KIND'] = None
                        else:
                            r[i]['KIND'] = None

                        new.append(r)
                out['data']['allInstances']['records'] = new

        logger.info("getAllInstances kn: {} kid: {}".format(kindName, kindId))
        self._check_response(out)
        return out["data"]

    async def getAllInstancesByName(self, kindName=None, fieldIds=None, take=0):
        try:
            k_id = await self.getKindID(kindName=kindName)
            return await self.getAllInstances(kindId=k_id, fieldIds=fieldIds, take=take)
        except Exception as e:
            logger.error(e)
            logger.error("Unable to get kind {} ".format(kindName))
            return None

    async def addInstance(self, addInstanceInput):
        query = string.Template("""
            mutation($tenantId: ID!, $addInstanceInput: AddInstanceInput!) {
                addInstance(tenantId: $tenantId, input: $addInstanceInput)
      }
        """)
        variables = {
            "tenantId": self.tenantId,
            "addInstanceInput": addInstanceInput
        }
        to_post = {
            "query": query.template,
            "variables": variables
        }
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()
        self._check_response(out)
        return out["data"]

    async def addInstanceByKindName(self, kindName, instance):
        try:
            kind = await self.getKind(kindId=None, kindName=kindName)
            imp = self._object_to_addInstanceInput(kind['kind'], instance)
            return await self.addInstance(imp)
        except Exception as e:
            logger.error(e)
            logger.error("Unable to get kind {} ".format(kindName))
            return None

    async def addInstanceByKindId(self, kindId, instance):
        try:
            kind = await self.getKind(kindId=kindId, kindName=None)
            imp = self._object_to_addInstanceInput(kind['kind'], instance)
            return await self.addInstance(imp)
        except Exception as e:
            logger.error(e)
            logger.error("Unable to get kind {} ".format(kindId))
            return None

    async def addInstanceSet(self, addInstanceSetInput):
        query = string.Template("""
          mutation($tenantId: ID!, $addInstanceSetInput: AddInstanceSetInput!) {
            addInstanceSet(tenantId: $tenantId, input: $addInstanceSetInput)
        }
        """)
        variables = {
            "tenantId": self.tenantId,
            "addInstanceSetInput": addInstanceSetInput
        }
        to_post = {
            "query": query.template,
            "variables": variables
        }
        resp = await self.session.post(self.svcUrl, data=json.dumps(to_post), headers=self.headers)
        out = await resp.json()
        self._check_response(out)
        return out["data"]

    async def addInstancesByKind(self, kind, instances):
        fieldIds, instanceIds, records = self._instanceSetFromObjects(kind['schema'], instances)
        addInstanceSetInput = {
            "kindId": kind['id'],
            "ids": instanceIds,
            "fieldIds": fieldIds,
            "records": records
        }

        return await self.addInstanceSet(addInstanceSetInput)

    async def addInstancesByKindName(self, kindName, instances):
        try:
            kind = await self.getKind(kindId=None, kindName=kindName)
            return await self.addInstancesByKind(kind['kind'], instances)
        except Exception as e:
            logger.error(e)
            logger.error("Unable to get kind {} ".format(kindName))
            return None

    async def addInstancesByKindId(self, kindId, instances):
        try:
            kind = await self.getKind(kindId=kindId, kindName=None)
            return await self.addInstancesByKind(kind['kind'], instances)
        except Exception as e:
            logger.error(e)
            logger.error("Unable to get kind {} ".format(kindId))
            return None