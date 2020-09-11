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
    anyone_wants_sizes = False
    for run in config['run'].values():
        for version in run.values():
            if version.get('errors'):
                anyone_wants_errors = True
            if version.get('colors'):
                anyone_wants_colors = True
            if version.get('sizes'):
                anyone_wants_sizes = True


    bands = config['bands']

    training_data = tc.load_data(
        config['training_file'],
        bands,
        errors=anyone_wants_errors,
        colors=anyone_wants_colors,
        size=anyone_wants_sizes
    )

    validation_data = tc.load_data(
        config['validation_file'],
        bands,
        errors=anyone_wants_errors,
        colors=anyone_wants_colors,
        size=anyone_wants_sizes
    )

    training_z = tc.load_redshift(config['training_file'])
    validation_z = tc.load_redshift(config['validation_file'])

    if config['metrics_impl'] == 'jax-cosmo':
        metrics_fn = tc.jc_compute_scores
    else:
        metrics_fn = tc.compute_scores

    with open(config['output_file'],'a') as output_file:
        for classifier_name, runs in config['run'].items():
            for run, settings in runs.items():
                scores = run_one(classifier_name, bands, settings,
                                 training_data, training_z, validation_data, validation_z,
                                 config['metrics'], metrics_fn)

                output_file.write (f"{classifier_name} {run} {settings} {scores} \n")



def run_one(classifier_name, bands, settings, train_data, train_z, valid_data,
             valid_z, metrics, metrics_fn):
    classifier = tc.Tomographer._find_subclass(classifier_name)

    if classifier.wants_arrays:
        errors = settings.get('errors')
        colors = settings.get('colors')
        sizes = settings.get('sizes')
        train_data = tc.dict_to_array(train_data, bands, errors=errors, colors=colors, sizes=sizes)
        valid_data = tc.dict_to_array(valid_data, bands, errors=errors, colors=colors, sizes=sizes)

    print ("Executing: ", classifier_name, bands, settings)

    ## first check if options are valid
    print (settings, classifier.valid_options)
    for key in settings.keys():
        if key not in classifier.valid_options and key not in ['errors', 'colors', 'sizes']:
            raise ValueError(f"Key {key} is not recognized by classifier {classifier_name}")

    print ("Initializing classifier...")
    C=classifier(bands, settings)

    print ("Training...")
    C.train(train_data,train_z)

    print ("Applying...")
    results = C.apply(valid_data)

    print ("Getting metric...")
    scores = metrics_fn(results, valid_z, metrics=metrics)

    print ("Making some pretty plots...")
    name = str(classifier.__name__)
    tc.metrics.plot_distributions(valid_z, results, f"plots/{name}_{bands}.png")

    return scores


if __name__=="__main__":
    main()
