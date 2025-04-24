def print_ucuu_hello(ending_words=None, *args, **kwargs):
    if ending_words is None:
        print("No Ending Words.")
    elif ending_words == "please raise errors":
        raise NotImplementedError("Raise Error due to requests")
    else:
        print(f"Hello, {ending_words}")
