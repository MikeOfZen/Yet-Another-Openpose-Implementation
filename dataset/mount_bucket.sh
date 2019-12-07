#!/bin/bash
mkdir TFrecords
gcsfuse --only-dir YAOP_TFrecords datasets_bucket_a ./TFrecords/
