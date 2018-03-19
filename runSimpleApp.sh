###
# Copyright 2016 MapR Technologies, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#  limitations under the License.
#
# Author: Dong Meng
# dmeng@mapr.com
###


#param="$1"
#if [ ! -n "$pparam" ] ; then
#	echo "ERROR: missing input param."
#	echo "USAGE: $0 input_param"
#	exit 1
#fi
DIR="$( cd "$( dirname "${BATH_SOURCE[0]}" )" && pwd )"

/opt/mapr/spark/spark-2.1.0/bin/spark-submit --master yarn --deploy-mode client --driver-memory 3g \
    --class "com.dmeng.xgboostexample.SimpleApp" ${DIR}/target/scala-2.11/xgboostsparkexample-assembly-0.1.0.jar \
    --param1 "content of this param" 
