import os
import time

import click
import googleapiclient
from recsys.automation.utils import get_timestamp


def wait_for_operation(compute, project, zone, operation):
    print('Waiting for operation to finish...')
    while True:
        result = compute.zoneOperations().get(
            project=project,
            zone=zone,
            operation=operation).execute()

        if result['status'] == 'DONE':
            print("done.")
            if 'error' in result:
                raise Exception(result['error'])
            return result

        time.sleep(1)


def clone_disk_from_snapshot(compute, project, zone, snapshot_name, disk_name):
    """
    POST
    https: // www.googleapis.com / compute / v1 / projects / [PROJECT_ID] / zones / [ZONE] / disks
    """

    config = {
        "name": disk_name,
        "sourceSnapshot": f"zones/{zone}/snapshots/{snapshot_name}"
    }

    return compute.disks().insert(
        project=project,
        zone=zone,
        body=config).execute()


def create_instance(compute, project, zone, name, disk_name, model_config, validation, storage_path):
    # Configure the machine
    machine_type = "zones/%s/machineTypes/n1-highmem-96" % zone
    startup_script = open(os.path.join(os.path.dirname(__file__), 'startup-script.sh'), 'r').read()
    config = {
        'name': name,
        'machineType': machine_type,

        # Specify the boot disk and the image to use as a source.
        'disks': [{
                "source": f"zones/{zone}/disks/{disk_name}",
                "boot": True
        }],

        # Specify a network interface with NAT to access the public
        # internet.
        'networkInterfaces': [{
            'network': 'global/networks/default',
            'accessConfigs': [
                {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
            ]
        }],

        # Allow the instance to access cloud storage and logging.
        'serviceAccounts': [{
            'email': 'default',
            'scopes': [
                'https://www.googleapis.com/auth/devstorage.read_write',
                'https://www.googleapis.com/auth/logging.write'
            ]
        }],

        # Metadata is readable from the instance and allows you to
        # pass configuration from deployment scripts to instances.
        'metadata': {
            'items': [{
                # Startup script is automatically executed by the
                # instance upon startup.
                'key': 'startup-script',
                'value': startup_script
            }, {
                'key': 'model_config',
                'value': model_config
            }, {
                'key': 'validation',
                'value': str(validation)
            }, {
                'key': 'storage_path',
                'value': storage_path
            }]
        }
    }

    return compute.instances().insert(
        project=project,
        zone=zone,
        body=config).execute()

def delete_instance(compute, project, zone, name):
    return compute.instances().delete(
        project=project,
        zone=zone,
        instance=name).execute()


@click.command()
@click.option("config_file", type=str)
@click.option("validation", type=bool)
def main(config_file, validation):
    compute = googleapiclient.discovery.build('compute', 'v1')
    project = "logicai-recsys2019"
    timestamp = get_timestamp()
    zone = "europe-west1-b"
    snapshot_name = "recsys1-models"
    instance_name = "recsys_tmp_{timestamp}"
    storage_path = "predictions/runs/{timestamp}/"
    disk_name = "recsys_tmp_disk_{timestamp}"
    with open(config_file) as inp:
        model_config = inp.read()
    validation = 1 if validation else 0

    print("Clone disk from snapshot")
    operation = clone_disk_from_snapshot(compute=compute,
                             project=project,
                             zone=zone,
                             snapshot_name=snapshot_name,
                             disk_name=disk_name)
    print("Waiting for creation")
    wait_for_operation(compute, project, zone, operation)
    print("Create instance")
    operation = create_instance(compute=compute,
                                project=project,
                                zone=zone,
                                name=instance_name,
                                model_config=model_config,
                                validation=validation,
                                storage_path=storage_path,
                                disk_name=disk_name)
    # print("Waiting for creation")
    # wait_for_operation(compute, project, zone, operation['name'])