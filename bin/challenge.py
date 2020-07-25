#!/usr/bin/env python
import sys
import os
import click
import yaml
import jinja2

## get root dir, one dir above this script
root_dir=os.path.join(os.path.split(sys.argv[0])[0],"..")
sys.path.append(root_dir)
import tomo_challenge as tc

from tomo_challenge.jax_metrics import compute_scores as jc_compute_scores

@click.command()
@click.argument('config_yaml', type=str)
def main(config_yaml):
    with open(config_yaml, 'r') as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)
    
    # Get the classes associated with each
    for name in config['run']:
        try:
            config['cls'] = tc.Tomographer._find_subclass(name)
        except KeyError:
            raise ValueError(f"Tomographer {name} is not defined")
    
    # Decide if anyone needs the colors calculating and/or errors loading
    anyone_wants_colors = False
    anyone_wants_errors = False
    for run in config['run'].values():
        for version in run.values():
            if version.get('errors'):
                anyone_wants_errors = True
            if version.get('colors'):
                anyone_wants_colors = True


    bands = config['bands']

    training_data = tc.load_data(
        config['training_file'],
        bands,
        errors=anyone_wants_errors,
        colors=anyone_wants_colors
    )

    validation_data = tc.load_data(
        config['validation_file'],
        bands,
        errors=anyone_wants_errors,
        colors=anyone_wants_colors
    )

    training_z = tc.load_redshift(config['training_file'])
    validation_z = tc.load_redshift(config['validation_file'])

    with open(config['output_file'],'w') as output_file:
        for classifier_name, runs in config['run'].items():
            for run, settings in runs.items():
                scores = run_one(classifier_name, bands, settings,
                                 training_data, training_z, validation_data, validation_z,
                                 config['metrics'])

                output_file.write (f"{classifier_name} {run} {settings} {scores} \n")
                print("scores= ",scores)


def run_one(classifier_name, bands, settings, train_data, train_z, valid_data,
             valid_z, metrics):
    classifier = tc.Tomographer._find_subclass(classifier_name)

    if classifier.wants_arrays:
        errors = settings.get('errors')
        colors = settings.get('colors')
        train_data = tc.dict_to_array(train_data, bands, errors=errors, colors=colors)
        valid_data = tc.dict_to_array(valid_data, bands, errors=errors, colors=colors)


        #JEC 23/7/2020 restrict  data
        train_data=train_data[:1000000,:]
        valid_data=valid_data[:50000,:]

        
    #JEC 23/7/2020 restrict  data
    train_z = train_z[:1000000]
    valid_z = valid_z[:50000]

    print ("Executing: ", classifier_name, bands, settings)

    ## first check if options are valid
    for key in settings.keys():
        if key not in classifier.valid_options and key not in ['errors', 'colors']:
            raise ValueError(f"Key {key} is not recognized by classifier {classifier_name}")

    print ("Initializing classifier...")
    C=classifier(bands, settings)

    print ("Training...")
    C.train(train_data,train_z)

    print ("Applying...")
    results = C.apply(valid_data)

    print ("Getting metric...")
    #    scores = tc.compute_scores(results, valid_z, metrics=metrics)
    #Use JAX code 23/7/2020
    scores = jc_compute_scores(results, valid_z, metrics="SNR_3x2,FOM_3x2,FOM_DETF_3x2")

    return scores


if __name__=="__main__":
    main()


