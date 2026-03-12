from src.env_vars import ALL_LINE_FILE_PATH


def read_line_file(line_file_path=ALL_LINE_FILE_PATH):
    """
        Read the line file and return author-file mappings.

        :param line_file_path: Path to the line file.
        :type line_file_path: str

        :return: data
        :rtype: tuple[dict[int, list[str]], list[int]]

            data: Mapping of author IDs to lists of file names.
            authors: Sorted list of unique author IDs found in the file.
        """
    data = dict()
    authors = []
    with open(line_file_path, "r", encoding="utf-8") as f:
        for line in f:
            file, author_id, *_ = line.split()
            authors.append(int(author_id))
            data[int(author_id)] = data.get(int(author_id), []) + [file]

    authors = list(set(authors))
    authors.sort()
    return data, authors
