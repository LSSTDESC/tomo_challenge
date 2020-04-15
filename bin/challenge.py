#!/usr/bin/env python
import sys
import os
import click
import json

## get root dir, one dir below this script
root_dir=os.path.join(os.path.split(sys.argv[0])[0],"..")
sys.path.append(root_dir)
import tomo_challenge as tc

@click.command()
@click.argument('action', type=str)
@click.option('-c', '--classifier', type=str, default='RandomForest', show_default=True, help = "comma separated classifier(s)")
@click.option('-b', '--bands', type=str, default="riz", show_default=True, help = "comma separated list of bands")
@click.option('-o', '--options', type=str, default='"bins":3', show_default=True, help ="comma separated list of option dics")
@click.option('-x', '--competition', is_flag=True, help = "run with competition catalog")
def main(action, classifier, bands, options, competition):
    classifiers = find_modules()
    print ("Found classifiers: ",", ".join(classifiers.keys()))
    
    if action == "one":
        if classifier not in classifiers:
            print (f"Classifier {classifier} not found.")
            sys.exit(1)
        opts = json.loads("{"+options+"}")
        run_one(classifier, classifiers[classifier], bands, opts, competition)


    
def find_modules():
    import glob, importlib.util
    classifiers = {}
    for file in glob.glob(os.path.join(root_dir,"classifiers","*.py")):
        spec = importlib.util.spec_from_file_location("", file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for item in dir(mod):
            obj = getattr(mod,item) 
            if hasattr(obj,"train") and hasattr(obj,"apply"):
                classifiers[item]= obj
    return classifiers
    

def run_one (id, classifier, bands, options, competition):
    print ("Loading data...")
    training_file = f'{bands}/training.hdf5'
    train_data = tc.load_magnitudes_and_colors(training_file, bands)
    train_z = tc.load_redshift(training_file)
    validation_file = f'{bands}/validation.hdf5'
    validation_data = tc.load_magnitudes_and_colors(validation_file, bands)
    validation_z = tc.load_redshift(validation_file)
    print ("Initializing classifier...")
    C=classifier(bands, options)
    print ("Training...")
    C.train(train_data,train_z)
    print ("Applying...")
    results = C.apply(validation_data)
    validation_redshift = tc.load_redshift(validation_file)
    print ("Getting metric...")
    scores = tc.compute_scores(results, validation_z)



if __name__=="__main__":
    main()


