#!/usr/bin/env python3

## Copyright 2023 Mark Thurston <mark.thurston@nhs.net>
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions
## are met:
##
## 1. Redistributions of source code must retain the above copyright
## notice, this list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above
## copyright notice, this list of conditions and the following
## disclaimer in the documentation and/or other materials provided
## with the distribution.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
## FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
## COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
## INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
## (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
## HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
## STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
## ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
## OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Create batches of images suitable for use with VIA VGG image labelling software

Allows distribution of different image batches to each image labelling team
member by just providing the project.json file
"""

import argparse
import json
import logging
import os
import random

###########################################################################
# config section: can be set here or on the command line
BATCH_SIZE = 75
IMG_PREFIX = ""
IMG_FILE_EXT = "jpg"
SHUFFLE_SEED = ""  # repeatable random shuffle if non-empty
TEMPLATE_PROJECT_JSON = "empty.json"
logging.basicConfig(level=logging.INFO)
INPUT_DIR = ""
OUTPUT_DIR = ""
###########################################################################


class ProjectFile:
    def __init__(self, fn: str, max_size: int):
        """
        Init involves loading the template JSON and ensuring it doesn't contain
        any images

        Use max_size = 0 to remove the image number limit
        """
        with open(TEMPLATE_PROJECT_JSON) as f:
            self.project_json = json.load(f)

        self.empty()

        # storage location for the project file
        self.output_fn = os.path.join(OUTPUT_DIR, fn)
        self.max_image_n = max_size

    def empty(self) -> None:
        """Remove all images from the project"""
        self.project_json["project"]["vid_list"] = []
        self.project_json["file"] = {}
        self.project_json["view"] = {}
        # number of images currently in the project
        self.n_images = 0

    def add(self, n: int, img_fn: str) -> None:
        """Add an extra image to the project"""
        # don't allow more than the maximum images in the project
        if self.n_images > self.max_image_n and self.max_image_n != 0:
            raise Exception(
                "Attempting to add more than the maximum ({}) to the project".format(
                    self.max_image_n
                )
            )

        str_n = str(n)
        # fail if n already exists in the project
        if (
            str_n in self.project_json["project"]["vid_list"]
            or str_n in self.project_json["file"]
            or str_n in self.project_json["view"]
        ):
            raise Exception(
                "Key ({}) already exists. Should be unique".format(n)
            )
        # update the internal JSON with the new image
        self.project_json["project"]["vid_list"].append(str_n)
        self.project_json["file"].update(
            {
                str_n: {
                    "fid": str_n,
                    "fname": IMG_PREFIX + img_fn,
                    "type": 2,
                    "loc": 3,
                    "src": IMG_PREFIX + img_fn,
                }
            }
        )
        self.project_json["view"].update({str_n: {"fid_list": [str_n]}})
        self.n_images += 1

    def save(self) -> None:
        """Save the project file to disk"""
        with open(self.output_fn, "w") as f:
            json.dump(self.project_json, f)


def create_image_list(input_dir: str = INPUT_DIR) -> list[str]:
    """Create a list of all available image files"""
    # directory containing all images for labelling
    all_input_images = [
        i
        for i in sorted(os.listdir(input_dir))
        if i.lower().endswith("." + IMG_FILE_EXT)
    ]
    # mix up the images
    if SHUFFLE_SEED:
        random.seed(SHUFFLE_SEED)
    random.shuffle(all_input_images)
    return all_input_images


def split_into_batches(
    full_list: list, batch_size: int = BATCH_SIZE
) -> list[list]:
    """Split the full list into smaller batches, max. length BATCH_SIZE"""
    n_full_batches = len(full_list) // BATCH_SIZE
    n_batches = n_full_batches + int(len(full_list) % BATCH_SIZE != 0)

    logging.info(
        "{} images identified ({} batches of {} (max))".format(
            len(full_list), n_batches, BATCH_SIZE
        )
    )

    r = []
    for i in range(n_batches):
        r.append(full_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])
        logging.info(
            "Batch {}: {} images".format(
                i, len(full_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])
            )
        )
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create template JSON files for all images contained within an"
            " image directory"
        )
    )
    parser.add_argument(
        "-n",
        "--batch-size",
        default=BATCH_SIZE,
        type=int,
        help="Number of images to include in a single batch",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help="Location of the image dataset (for input)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Location to store the created project files",
    )
    parser.add_argument(
        "-p",
        "--image-prefix",
        type=str,
        default=IMG_PREFIX,
        help="Prefix the image paths (e.g. full or relative URL)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=str,
        default=SHUFFLE_SEED,
        help=(
            "Random seed.  If provided, image order will be"
            " deterministic/repeatable"
        ),
    )
    args = parser.parse_args()

    # take into account the command line arguments
    BATCH_SIZE = args.batch_size
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    IMG_PREFIX = args.image_prefix

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = create_image_list(INPUT_DIR)

    for i, j in enumerate(split_into_batches(images)):
        logging.info("Processing batch {}".format(i))
        # leading zeros allow sorting of the files by filename
        p = ProjectFile(fn="{:03d}.json".format(i), max_size=BATCH_SIZE)
        for img_n, img in enumerate(j):
            # combine the batch number and the image number to ensure unique
            # identifiers for all images (even across different batches)
            p.add(BATCH_SIZE * i + img_n, str(img))
            p.save()
