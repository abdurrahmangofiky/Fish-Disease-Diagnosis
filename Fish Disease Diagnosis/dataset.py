from bing_image_downloader import downloader

labels = [
    "aeromonas fish disease",
    "saprolegnia fungus fish",
    "koi herpesvirus",
    "parasitic fish disease",
    "ichthyophonus fish",
    "gill necrosis fish"
]

for label in labels:
    downloader.download(
        query=label,
        limit=80,            
        output_dir='dataset_raw',
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )
