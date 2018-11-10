# 20 Newsgroups

See [sample pipeline](sample_scripts/20_newsgroups.py) and [sample webservice](sample_scripts/webservice.py).

To run the sample code:

    cd sample_scripts
    python 20_newsgroups.py # model selection
    python webservice.py # starts HTTP service

To query the web service with the selected model:

    curl http://localhost:5000 -H "Content-Type: application/json" -d '{"post_text":"how to build a battery"}' -X POST

# Amazon Product data

A set of sample scripts can be found [here](amazon-product-data/), find the data that is used [here](http://jmcauley.ucsd.edu/data/amazon).

# Amazon Customer Reviews Dataset

See [this link](https://registry.opendata.aws/amazon-reviews) for details of the dataset. Documentation of the dataset can be found [here](https://s3.amazonaws.com/amazon-reviews-pds/readme.html).

