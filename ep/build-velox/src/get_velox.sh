#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -exu

VELOX_REPO=https://github.com/oap-project/velox.git
VELOX_BRANCH=2025_07_31
VELOX_HOME=""
RUN_SETUP_SCRIPT=ON
VELOX_ENHANCED_REPO=https://github.com/IBM/velox.git
VELOX_ENHANCED_BRANCH=ibm-2025_07_31
ENABLE_ENHANCED_FEATURES=OFF

# Developer use only for testing Velox PR.
UPSTREAM_VELOX_PR_ID=""

OS=`uname -s`

for arg in "$@"; do
  case $arg in
  --velox_repo=*)
    VELOX_REPO=("${arg#*=}")
    VELOX_ENHANCED_REPO=("${arg#*=}")
    shift # Remove argument name from processing
    ;;
  --velox_branch=*)
    VELOX_BRANCH=("${arg#*=}")
    VELOX_ENHANCED_BRANCH=("${arg#*=}")
    shift # Remove argument name from processing
    ;;
  --velox_home=*)
    VELOX_HOME=("${arg#*=}")
    shift # Remove argument name from processing
    ;;
  --run_setup_script=*)
    RUN_SETUP_SCRIPT=("${arg#*=}")
    shift # Remove argument name from processing
    ;;
  --enable_enhanced_features=*)
    ENABLE_ENHANCED_FEATURES=("${arg#*=}")
    VELOX_REPO=$VELOX_ENHANCED_REPO
    VELOX_BRANCH=$VELOX_ENHANCED_BRANCH
    shift # Remove argument name from processing
    ;;
  *)
    OTHER_ARGUMENTS+=("$1")
    shift # Remove generic argument from processing
    ;;
  esac
done

function ensure_pattern_matched {
  if [ $# -ne 2 ]; then
    echo "Exactly 2 arguments are required."
    return 1
  fi
  pattern=$1
  file=$2
  matched_lines=$(grep -c "$pattern" $file)
  if [ $matched_lines -eq 0 ]; then
    return 1
  fi
}

function process_setup_ubuntu {
  echo "Using setup script from Velox"
}

function process_setup_centos9 {
  echo "Using setup script from Velox"
}

function process_setup_alinux3 {
  sed -i "/^[[:space:]]*#/!s/.*dnf_install epel-release/#&/" ${CURRENT_DIR}/setup-centos8.sh
  sed -i "/^[[:space:]]*#/!s/.*run_and_time install_conda/#&/" ${CURRENT_DIR}/setup-centos8.sh
  sed -i "/^[[:space:]]*#/!s/.*dnf config-manager --set-enabled powertools/#&/" ${CURRENT_DIR}/setup-centos8.sh
  sed -i "s/gcc-toolset-11 //" ${CURRENT_DIR}/setup-centos8.sh
  sed -i "/^[[:space:]]*#/!s/.*source \/opt\/rh\/gcc-toolset-11\/enable/#&/" ${CURRENT_DIR}/setup-centos8.sh
  sed -i "/^[[:space:]]*#/!s|^export CC=/opt/rh/gcc-toolset-11/root/bin/gcc|# &|" ${CURRENT_DIR}/setup-centos8.sh
  sed -i "/^[[:space:]]*#/!s|^export CXX=/opt/rh/gcc-toolset-11/root/bin/g++|# &|" ${CURRENT_DIR}/setup-centos8.sh
  sed -i 's/python39 python39-devel python39-pip //g' ${CURRENT_DIR}/setup-centos8.sh
  sed -i "/^[[:space:]]*#/!s/.*pip.* install/#&/" ${CURRENT_DIR}/setup-centos8.sh
}

function process_setup_tencentos32 {
  sed -i "/^[[:space:]]*#/!s/.*dnf config-manager --set-enabled powertools/#&/" ${CURRENT_DIR}/setup-centos8.sh
}

echo "Preparing Velox source code..."

CURRENT_DIR=$(
  cd "$(dirname "$BASH_SOURCE")"
  pwd
)

if [ "$VELOX_HOME" == "" ]; then
  VELOX_HOME="$CURRENT_DIR/../build/velox_ep"
fi
VELOX_SOURCE_DIR="${VELOX_HOME}"

# checkout code
TARGET_BUILD_COMMIT="$(git ls-remote $VELOX_REPO $VELOX_BRANCH | awk '{print $1;}' | head -n 1)"
if [ -d $VELOX_SOURCE_DIR ]; then
  echo "Velox source folder $VELOX_SOURCE_DIR already exists..."
  cd $VELOX_SOURCE_DIR
  # if velox_branch exists, check it out, 
  # otherwise assume that user prepared velox source in velox_home, skip checkout
  if [ -n "$TARGET_BUILD_COMMIT" ]; then
    git init .
    EXISTS=$(git show-ref refs/tags/build_$TARGET_BUILD_COMMIT || true)
    if [ -z "$EXISTS" ]; then
      git fetch $VELOX_REPO $TARGET_BUILD_COMMIT:refs/tags/build_$TARGET_BUILD_COMMIT
    fi
    git reset --hard HEAD
    git checkout refs/tags/build_$TARGET_BUILD_COMMIT
  else
    echo "$VELOX_BRANCH can't be found in $VELOX_REPO, skipping the download..."
  fi
else
  git clone $VELOX_REPO -b $VELOX_BRANCH $VELOX_SOURCE_DIR
  cd $VELOX_SOURCE_DIR
  git checkout $TARGET_BUILD_COMMIT
fi

#sync submodules
git submodule sync --recursive
git submodule update --init --recursive

function apply_provided_velox_patch {
  if [[ -n "$UPSTREAM_VELOX_PR_ID" ]]; then
     echo "Applying patch for PR #$UPSTREAM_VELOX_PR_ID ..."
     local velox_home=$1
     local patch_name="$UPSTREAM_VELOX_PR_ID.patch"
     pushd $velox_home
     rm -f $patch_name
     wget -nv "https://patch-diff.githubusercontent.com/raw/facebookincubator/velox/pull/$UPSTREAM_VELOX_PR_ID.patch" \
       -O "$patch_name" || {
       echo "Failed to download the Velox patch from GitHub"
       exit 1
     }
     (git apply --check $patch_name && git apply $patch_name) || {
       echo "Failed to apply the provided Velox patch"
       exit 1
     }
     popd
  fi
}

function apply_compilation_fixes {
  current_dir=$1
  velox_home=$2

  sudo cp ${current_dir}/modify_arrow.patch ${velox_home}/CMake/resolve_dependency_modules/arrow/
  sudo cp ${current_dir}/modify_arrow_dataset_scan_option.patch ${velox_home}/CMake/resolve_dependency_modules/arrow/

  git add ${velox_home}/CMake/resolve_dependency_modules/arrow/modify_arrow.patch # to avoid the file from being deleted by git clean -dffx :/
  git add ${velox_home}/CMake/resolve_dependency_modules/arrow/modify_arrow_dataset_scan_option.patch # to avoid the file from being deleted by git clean -dffx :/

}

function setup_linux {
  local LINUX_DISTRIBUTION=$(. /etc/os-release && echo ${ID})
  local LINUX_VERSION_ID=$(. /etc/os-release && echo ${VERSION_ID})

  export SUDO="sudo --preserve-env"
  if [[ "$LINUX_DISTRIBUTION" == "ubuntu" || "$LINUX_DISTRIBUTION" == "debian" || "$LINUX_DISTRIBUTION" == "pop" ]]; then
    process_setup_ubuntu
  elif [[ "$LINUX_DISTRIBUTION" == "centos" ]]; then
    case "$LINUX_VERSION_ID" in
      9) process_setup_centos9 ;;
      8) ;;
      7) ;;
      *)
        echo "Unsupported centos version: $LINUX_VERSION_ID"
        exit 1
      ;;
    esac
  elif [[ "$LINUX_DISTRIBUTION" == "openEuler" ]]; then
    case "$LINUX_VERSION_ID" in
      24.03) ;;
      *)
        echo "Unsupported openEuler version: $LINUX_VERSION_ID"
        exit 1
      ;;
    esac
  elif [[ "$LINUX_DISTRIBUTION" == "alinux" ]]; then
    case "${LINUX_VERSION_ID:0:1}" in
      2) ;;
      3) process_setup_alinux3 ;;
      *)
        echo "Unsupported alinux version: $LINUX_VERSION_ID"
        exit 1
      ;;
    esac
  elif [[ "$LINUX_DISTRIBUTION" == "tencentos" ]]; then
    case "$LINUX_VERSION_ID" in
      2.4) ;;
      3.2) process_setup_tencentos32 ;;
      *)
        echo "Unsupported tencentos version: $LINUX_VERSION_ID"
        exit 1
      ;;
    esac
  elif [[ "$LINUX_DISTRIBUTION" == "rhel" ]]; then
    case "$LINUX_VERSION_ID" in
      9.6) ;;
      *)
        echo "Unsupported openEuler version: $LINUX_VERSION_ID"
        exit 1
      ;;
    esac
  else
    echo "Unsupported linux distribution: $LINUX_DISTRIBUTION"
    exit 1
  fi
}

apply_provided_velox_patch $VELOX_SOURCE_DIR

if [[ "$RUN_SETUP_SCRIPT" == "ON" ]]; then
  if [ $OS == 'Linux' ]; then
    setup_linux
  elif [ $OS == 'Darwin' ]; then
    :
  else
    echo "Unsupported kernel: $OS"
    exit 1
  fi
fi

apply_compilation_fixes $CURRENT_DIR $VELOX_SOURCE_DIR

echo "Finished getting Velox code"
