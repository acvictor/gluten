#! /bin/bash

set -e

BUILD_TESTS=OFF
ENABLE_S3=OFF
ENABLE_GCS=OFF
ENABLE_HDFS=OFF
ENABLE_ABFS=OFF

for arg in "$@"; do
  case $arg in
  --build_tests=*)
    BUILD_TESTS=("${arg#*=}")
    shift # Remove argument name from processing
    ;;
  --enable_s3=*)
    ENABLE_S3=("${arg#*=}")
    shift # Remove argument name from processing
    ;;
  --enable_gcs=*)
    ENABLE_GCS=("${arg#*=}")
    shift # Remove argument name from processing
    ;;
  --enable_hdfs=*)
    ENABLE_HDFS=("${arg#*=}")
    shift # Remove argument name from processing
    ;;
  --enable_abfs=*)
    ENABLE_ABFS=("${arg#*=}")
    shift # Remove argument name from processing
    ;;
  *)
    echo "Unrecognized argument: $arg"
    exit 1
    ;;
  esac
done

require_set() {
  if [ -z "${!1}" ]; then
    echo "Required variable $1 not found!"
    exit 1
  fi
}

require_set "VCPKG_ROOT"
require_set "VCPKG"
require_set "VCPKG_TRIPLET"
require_set "VCPKG_TRIPLET_INSTALL_DIR"

SCRIPT_ROOT="$(realpath "$(dirname "$0")")"
cd "$SCRIPT_ROOT"

if [ ! -d "$VCPKG_ROOT" ] || [ -z "$(ls "$VCPKG_ROOT")" ]; then
    git clone https://github.com/microsoft/vcpkg.git --branch 2023.10.19 "$VCPKG_ROOT"
fi
[ -f "$VCPKG" ] || "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics

EXTRA_FEATURES=""
if [ "$BUILD_TESTS" = "ON" ]; then
  EXTRA_FEATURES+="--x-feature=duckdb "
fi
if [ "$ENABLE_S3" = "ON" ]; then
  EXTRA_FEATURES+="--x-feature=velox-s3 "
fi
if [ "$ENABLE_GCS" = "ON" ]; then
  EXTRA_FEATURES+="--x-feature=velox-gcs "
fi
if [ "$ENABLE_HDFS" = "ON" ]; then
  EXTRA_FEATURES+="--x-feature=velox-hdfs "
fi
if [ "$ENABLE_ABFS" = "ON" ]; then
  EXTRA_FEATURES+="--x-feature=velox-abfs"
fi


$VCPKG install --no-print-usage \
    --triplet="${VCPKG_TRIPLET}" --host-triplet="${VCPKG_TRIPLET}" ${EXTRA_FEATURES}

# For fixing a build error like below when gluten's build type is Debug:
# No rule to make target '/root/gluten/dev/vcpkg/vcpkg_installed/x64-linux-avx/debug/lib/libz.a',
# needed by 'releases/libvelox.so'
mkdir -p $VCPKG_TRIPLET_INSTALL_DIR/debug/lib/
cp $VCPKG_TRIPLET_INSTALL_DIR/lib/libz.a $VCPKG_TRIPLET_INSTALL_DIR/debug/lib
cp $VCPKG_TRIPLET_INSTALL_DIR/lib/libssl.a $VCPKG_TRIPLET_INSTALL_DIR/debug/lib
cp $VCPKG_TRIPLET_INSTALL_DIR/lib/libcrypto.a $VCPKG_TRIPLET_INSTALL_DIR/debug/lib
cp $VCPKG_TRIPLET_INSTALL_DIR/lib/liblzma.a $VCPKG_TRIPLET_INSTALL_DIR/debug/lib
cp $VCPKG_TRIPLET_INSTALL_DIR/lib/libdwarf.a $VCPKG_TRIPLET_INSTALL_DIR/debug/lib
# Allow libhdfs3.a is not installed as build option may not enable hdfs.
cp $VCPKG_TRIPLET_INSTALL_DIR/lib/libhdfs3.a $VCPKG_TRIPLET_INSTALL_DIR/debug/lib || true

