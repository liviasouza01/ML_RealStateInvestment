import nltk
import ssl

def download_resources():
    """
    Downloads the necessary NLTK resources for the project,
    handling potential SSL errors.
    """
    resources = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'punkt_tab': 'tokenizers/punkt_tab'
    }
    
    print("Checking for NLTK resources...")

    # HACK: This is a workaround for SSL certificate verification errors.
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    all_good = True
    for resource_id, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
            print(f"✅ '{resource_id}' is already downloaded.")
        except LookupError:
            print(f"⚠️ '{resource_id}' not found. Downloading now...")
            all_good = False
            try:
                nltk.download(resource_id)
            except Exception as e:
                print(f"❌ Failed to download '{resource_id}': {e}")
                print("Please check your internet connection and try again.")
                print("You might also need to run this with administrator/sudo privileges.")
                return

    if all_good:
        print("\nAll necessary NLTK resources are available.")
    else:
        print("\nNLTK resource download process finished.")

if __name__ == "__main__":
    download_resources() 