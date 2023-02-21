# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Various utilities specific to data processing.
"""
import collections
import logging
import os
import re
import sys
import tarfile
import urllib.request
import zipfile
from typing import Any, List, Optional, overload, Union, Dict, Tuple

import numpy as np

__all__ = [
    "maybe_download",
    "read_words",
    "make_vocab",
    "count_file_lines",
    "get_filename"
]

Py3 = sys.version_info[0] == 3


def maybe_download(
    urls: Union[List[str], str],
    path: Union[str, os.PathLike],
    filenames: Union[List[str], str, None] = None,
    extract: bool = False,
    num_gdrive_retries: int = 1,
):
    r"""Downloads a set of files.
    Args:
        urls: A (list of) URLs to download files.
        path: The destination path to save the files.
        filenames: A (list of) strings of the file names. If given,
            must have the same length with ``urls``. If `None`,
            filenames are extracted from ``urls``.
        extract: Whether to extract compressed files.
        num_gdrive_retries: An integer specifying the number of attempts
            to download file from Google Drive. Default value is 1.
    Returns:
        A list of paths to the downloaded files.
    """
    maybe_create_dir(path)

    if not isinstance(urls, (list, tuple)):
        is_list = False
        urls = [urls]
    else:
        is_list = True
    if filenames is not None:
        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]
        if len(urls) != len(filenames):
            raise ValueError(
                "`filenames` must have the same number of elements as `urls`."
            )

    result = []
    for i, url in enumerate(urls):
        if filenames is not None:
            filename = filenames[i]
        elif "drive.google.com" in url:
            filename = _extract_google_drive_file_id(url)
        else:
            filename = url.split("/")[-1]
            # If downloading from GitHub, remove suffix ?raw=True
            # from local filename
            if filename.endswith("?raw=true"):
                filename = filename[:-9]

        filepath = os.path.join(path, filename)
        result.append(filepath)

        # if not tf.gfile.Exists(filepath):
        if not os.path.exists(filepath):
            if "drive.google.com" in url:
                filepath = _download_from_google_drive(
                    url, filename, path, num_gdrive_retries
                )
            else:
                filepath = _download(url, filename, path)

            if extract:
                logging.info("Extract %s", filepath)
                if tarfile.is_tarfile(filepath):
                    with tarfile.open(filepath, "r") as tfile:
                        def is_within_directory(directory, target):
                            
                            abs_directory = os.path.abspath(directory)
                            abs_target = os.path.abspath(target)
                        
                            prefix = os.path.commonprefix([abs_directory, abs_target])
                            
                            return prefix == abs_directory
                        
                        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                        
                            for member in tar.getmembers():
                                member_path = os.path.join(path, member.name)
                                if not is_within_directory(path, member_path):
                                    raise Exception("Attempted Path Traversal in Tar File")
                        
                            tar.extractall(path, members, numeric_owner=numeric_owner) 
                            
                        
                        safe_extract(tfile, path)
                elif zipfile.is_zipfile(filepath):
                    with zipfile.ZipFile(filepath) as zfile:
                        zfile.extractall(path)
                else:
                    logging.info(
                        "Unknown compression type. Only .tar.gz"
                        ".tar.bz2, .tar, and .zip are supported"
                    )
    if not is_list:
        return result[0]
    return result


# pylint: enable=unused-argument,function-redefined,missing-docstring


def _download(url: str, filename: str, path: Union[os.PathLike, str]) -> str:
    def _progress_hook(count, block_size, total_size):
        percent = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write(f"\r>> Downloading {filename} {percent:.1f}%")
        sys.stdout.flush()

    filepath = os.path.join(path, filename)
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress_hook)
    print()
    statinfo = os.stat(filepath)
    logging.info(
        "Successfully downloaded %s %d bytes", filename, statinfo.st_size
    )

    return filepath


def _extract_google_drive_file_id(url: str) -> str:
    # id is between `/d/` and '/'
    url_suffix = url[url.find("/d/") + 3 :]
    if url_suffix.find("/") == -1:
        # if there's no trailing '/'
        return url_suffix
    file_id = url_suffix[: url_suffix.find("/")]
    return file_id


def _download_from_google_drive(
    url: str, filename: str, path: Union[str, os.PathLike], num_retries: int = 1
) -> str:
    r"""Adapted from `https://github.com/saurabhshri/gdrive-downloader`"""

    # pylint: disable=import-outside-toplevel
    try:
        import requests
        from requests import HTTPError
    except ImportError:
        logging.info(
            "The requests library must be installed to download files from "
            "Google drive. Please see: https://github.com/psf/requests"
        )
        raise

    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        if "Google Drive - Virus scan warning" in response.text:
            match = re.search("confirm=([0-9A-Za-z_]+)", response.text)
            if match is None or len(match.groups()) < 1:
                raise ValueError(
                    "No token found in warning page from Google Drive."
                )
            return match.groups()[0]
        return None

    file_id = _extract_google_drive_file_id(url)

    gurl = "https://docs.google.com/uc?export=download"
    sess = requests.Session()
    params = {"id": file_id}
    response = sess.get(gurl, params=params, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = sess.get(gurl, params=params, stream=True)
    while response.status_code != 200 and num_retries > 0:
        response = requests.get(gurl, params=params, stream=True)
        num_retries -= 1
    if response.status_code != 200:
        logging.error(
            "Failed to download %s because of invalid response "
            "from %s: status_code='%d' reason='%s' content=%s, If you see this error message multiple times, you can download it directly. The links are saved in ./ctc_score/config.py. Put the downloaded files in the ~/.cache/ctc_score_models/_dataset_name_/ folder",
            filename,
            response.url,
            response.status_code,
            response.reason,
            response.content,
        )
        raise HTTPError(response=response)

    filepath = os.path.join(path, filename)
    CHUNK_SIZE = 32768
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    logging.info("Successfully downloaded %s", filename)

    return filepath


def read_words(filename: str, newline_token: Optional[str] = None) -> List[str]:
    r"""Reads word from a file.

    Args:
        filename (str): Path to the file.
        newline_token (str, optional): The token to replace the original newline
            token "\\n". For example, :python:`tx.data.SpecialTokens.EOS`.
            If `None`, no replacement is performed.

    Returns:
        A list of words.
    """
    with open(filename, "r") as f:
        if Py3:
            if newline_token is None:
                return f.read().split()
            else:
                return f.read().replace("\n", newline_token).split()
        else:
            if newline_token is None:
                return f.read().split()
            else:
                return f.read().replace("\n", newline_token).split()


def make_vocab(filenames, max_vocab_size=-1, newline_token=None,
               return_type="list", return_count=False):
    r"""Builds vocab of the files.

    Args:
        filenames (str): A (list of) files.
        max_vocab_size (int): Maximum size of the vocabulary. Low frequency
            words that exceeding the limit will be discarded.
            Set to `-1` (default) if no truncation is wanted.
        newline_token (str, optional): The token to replace the original newline
            token "\\n". For example, :python:`tx.data.SpecialTokens.EOS`.
            If `None`, no replacement is performed.
        return_type (str): Either ``list`` or ``dict``. If ``list`` (default),
            this function returns a list of words sorted by frequency. If
            ``dict``, this function returns a dict mapping words to their index
            sorted by frequency.
        return_count (bool): Whether to return word counts. If `True` and
            :attr:`return_type` is ``dict``, then a count dict is returned,
            which is a mapping from words to their frequency.

    Returns:
        - If :attr:`return_count` is False, returns a list or dict containing
          the vocabulary words.

        - If :attr:`return_count` if True, returns a pair of list or dict
          `(a, b)`, where `a` is a list or dict containing the vocabulary
          words, `b` is a list or dict containing the word counts.
    """

    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]

    words: List[str] = []
    for fn in filenames:
        words += read_words(fn, newline_token=newline_token)

    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, counts = list(zip(*count_pairs))
    words: List[str]
    counts: List[int]
    if max_vocab_size >= 0:
        words = words[:max_vocab_size]
    counts = counts[:max_vocab_size]

    if return_type == "list":
        if not return_count:
            return words
        else:
            return words, counts
    elif return_type == "dict":
        word_to_id = dict(zip(words, range(len(words))))
        if not return_count:
            return word_to_id
        else:
            word_to_count = dict(zip(words, counts))
            return word_to_id, word_to_count
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


# pylint: enable=unused-argument,function-redefined,missing-docstring

def count_file_lines(filenames: Any) -> int:
    r"""Counts the number of lines in the file(s).
    """

    def _count_lines(fn):
        with open(fn, "rb") as f:
            i = -1
            for i, _ in enumerate(f):
                pass
            return i + 1

    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]
    num_lines = np.sum([_count_lines(fn) for fn in filenames]).item()
    return num_lines


def get_filename(url: str) -> str:
    r"""Extracts the filename of the downloaded checkpoint file from the URL.
    """
    if 'drive.google.com' in url:
        return _extract_google_drive_file_id(url)
    url, filename = os.path.split(url)
    return filename or os.path.basename(url)


def maybe_create_dir(dirname: str) -> bool:
    r"""Creates directory if it does not exist.

    Args:
        dirname (str): Path to the directory.

    Returns:
        bool: Whether a new directory is created.
    """
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        return True
    return False


if __name__ == "__main__":
    cons_data_dict = {
        'xsum': 'https://drive.google.com/file/d/1_LuXfxT24ysZqrmpnqa83DV4g8_s8obR/view?usp=sharing',
        'cnndm': 'https://drive.google.com/file/d/1OqtILxhnXaC0OPpO9g00KX222JPfwoFb/view?usp=sharing',
        'cnndm_ref': 'https://drive.google.com/file/d/1xvKT_w50s3CyIzokW7WRMj9dOzzxh01E/view?usp=sharing',
        'yelp': 'https://drive.google.com/file/d/1BnjEekhSwQs5xovsWOrRbJabNo1u1nIg/view?usp=sharing',
        'persona_chat': 'https://drive.google.com/file/d/1Gpo56jXOdCZeiJjWc6NHTzcFCny2b382/view?usp=sharing',
        'persona_chat_fact': 'https://drive.google.com/file/d/1xj2xTj5ADDD8b7Dh7ZnGCXj6lLL3hJjU/view?usp=sharing',
        'topical_chat': 'https://drive.google.com/file/d/1bAIE99JFo1d6DWuR35uvyTe0d4_aZ6zb/view?usp=sharing',
        'topical_chat_fact': 'https://drive.google.com/file/d/1zI8TreDQOfadGMTmR2GOP8Ghh32uOrl5/view?usp=sharing'
    }

    for (dataset_name, dataset_link) in cons_data_dict.items():
        os.makedirs(
            f'./train/constructed_data/{dataset_name}/',
            exist_ok=True)
        maybe_download(
        urls=dataset_link,
        path= f'./train/constructed_data/{dataset_name}/',
        filenames=f'example.json')
