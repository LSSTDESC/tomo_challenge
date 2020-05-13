#!/usr/bin/env python
import sys
import os
import click
import yaml
import jinja2

## get root dir, one dir below this script
root_dir=os.path.join(os.path.split(sys.argv[0])[0],"..")
sys.path.append(root_dir)
import tomo_challenge as tc

@click.command()
@click.argument('config_yaml', type=str)

def main(config_yaml):
    with open(config_yaml, 'r') as fp:
        config_str = jinja2.Template(fp.read()).render()
    config = yaml.load(config_str, Loader=yaml.Loader)

    classifiers = find_modules()
    print ("Found classifiers: ",", ".join(classifiers.keys()))
    ## First check if classifiers are there
    for item in config['run']:
        if item not in classifiers:
            print (f'Classifier {item} not found.')
            raise NotImplementedError
    bands = config['bands']
    print ("Loading data...")
    training_data, training_z = load_data (config['training_file'], bands)
    validation_data, validation_z = load_data (config['validation_file'], bands)
    of = open(config['output_file'],'w')
    for classifier, runs in config['run'].items():
        for run, settings in runs.items():
            print ("Executing: ", classifier, bands, settings)
            scores = run_one(classifier, classifiers[classifier], bands, settings,
                             training_data, training_z, validation_data, validation_z,
                             config['metrics'])
            of.write (f"{classifier} {run} {settings} {scores} \n")
    of.close()

def find_modules():
    import glob, importlib.util
    classifiers = {}
    # Find every class in every module in the classifiers subdirectory that has a
    # method called train and another called apply.
    for file in glob.glob(os.path.join(root_dir,"classifiers","*.py")):
        spec = importlib.util.spec_from_file_location("", file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for item in dir(mod):
            obj = getattr(mod,item)
            if (hasattr(obj,"train") and hasattr(obj,"apply")
                and hasattr(obj,"valid_options") ):
                classifiers[item]= obj
    return classifiers

def load_data(fname, bands):
    data = tc.load_magnitudes_and_colors(fname, bands)
    Nbands = len(bands)
    print ('max mag before:',data[:,:Nbands].max(axis=0), 'nobj:',data.shape[0])
    z = tc.load_redshift(fname)
    data, z = tc.add_noise_snr_cut(data, z,  bands)
    print ('max mag after:',data[:,:Nbands].max(axis=0), 'nobj:',data.shape[0])

    return data,z

def run_one (id, classifier, bands, set, train_data, train_z, valid_data,
             valid_z, metrics):
    ## first check if options are valid
    for key in set.keys():
        if key not in classifier.valid_options:
            print ("Key %s is not recognized by classifier %s."%(key,id))
            raise NotImplementedError
    print ("Initializing classifier...")
    C=classifier(bands, set)
    print ("Training...")
    C.train(train_data,train_z)
    print ("Applying...")
    results = C.apply(valid_data)
    print ("Getting metric...")
    scores = tc.compute_scores(results, valid_z, metrics=metrics)
    return scores


if __name__=="__main__":
    main()
