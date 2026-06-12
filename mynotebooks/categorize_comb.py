#!/usr/bin/env python3
"""
Categorize image description texts + ImageNet-1K classes into groups.
Each item can belong to multiple groups.
Multi-property assignment: items are assigned not only by primary category
but also by salient visual properties (color, texture, pattern, shape).
"""

import json
import re
from collections import defaultdict, OrderedDict

# ─── Read texts ───────────────────────────────────────────────────────────────
with open('/home/nfm/clip_text_span/text_descriptions/image_descriptions_general.txt', 'r') as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"Total description texts loaded: {len(texts)}")

# ─── Read ImageNet classes ────────────────────────────────────────────────────
imagenet_classes = []
with open('/home/nfm/ViT-Prisma/mynotebooks/imagenet_classes.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Format: "idx: class_name"
        parts = line.split(': ', 1)
        if len(parts) == 2:
            cls_name = parts[1].strip()
            imagenet_classes.append(cls_name)

print(f"Total ImageNet classes loaded: {len(imagenet_classes)}")

# Combine: we tag each item with its source
all_items = []
for t in texts:
    all_items.append(("text", t))
for c in imagenet_classes:
    all_items.append(("imagenet", c))

print(f"Total items to categorize: {len(all_items)}")

# ─── Matching helper ──────────────────────────────────────────────────────────
def normalize(s):
    """Lowercase and replace underscores with spaces for matching."""
    return s.lower().replace('_', ' ')

def matches(item_normalized, keywords):
    return any(kw in item_normalized for kw in keywords)

# ─── Group definitions ────────────────────────────────────────────────────────
# Each group: list of lowercase substrings (any match → item belongs to group)
# ImageNet class names have underscores replaced with spaces before matching.

GROUPS = OrderedDict([

    # ══════════════════════════════════════════════════════════════════════════
    # VISUAL / PHOTOGRAPHIC PROPERTIES
    # ══════════════════════════════════════════════════════════════════════════

    ("Colors", [
        "color", "colour", " hue", "palette",
        "colorful", "monochromat", "sepia", "grayscale",
        "chromatic", "saturated", "desaturated", "pastel",
        "warm tone", "cool tone", "muted tone", "earthy tone",
        "soft tone", "vivid color", "earthy color",
        "burst of color", "pop art color", "high saturation",
        "minimal color", "evolving color", "picture of colors",
        "image with colors", "image with a complementary color",
        "image with a contrasting color", "image with a cool color",
        "image with a warm color", "image with a muted color",
        "image with a high-contrast color", "image with a monochromatic color",
        "image with a harmonious color", "image with a pastel color",
        "image with a single dominant color", "image with a variety of colors",
        "image with a vibrant color", "image with a gradient of colors",
        "photo with bold, contrasting tone", "photo with calming, pastel tone",
        "photo with cool, misty tone", "photo with muted, desaturated tone",
        "photo with vibrant, contrasting color", "photo with vibrant, saturated color",
        "photo with warm, golden", "photo with soft, pastel",
        "photo with soft, muted tone", "photo with soft, dreamy tone",
        "photo with faded, nostalgic color", "photograph with a blue color",
        "photograph with a brown color", "photograph with a green color",
        "photograph with a purple color", "photograph with a red color",
        "photograph with a yellow color",
        "a charcoal gray", "a gold color", "a grey color",
        "a silver color", "a platinum silver", "an amber color",
        "image with a black color", "image with a blue color",
        "image with a brown color", "image with a gray color",
        "image with a green color", "image with a orange color",
        "image with a pink color", "image with a purple color",
        "image with a red color", "image with a white color",
        "image with a yellow color", "playful color", "playful hue",
        "vibrant color", "psychedelic color", "harmonious color scheme",
        # ── ImageNet classes where color is a salient visual property ──
        # Greens
        "green lizard", "green snake", "green mamba",
        "granny smith",  # green apple
        # Golds / Yellows
        "goldfish", "goldfinch", "golden retriever",
        "sulphur butterfly", "sulphur-crested cockatoo",
        "yellow lady", "banana", "lemon", "corn",
        # Reds / Oranges
        "red fox", "red wolf", "redbone",
        "red-breasted merganser", "red-backed sandpiper",
        "red wine", "fire engine", "ladybug", "robin",
        "strawberry", "orangutan", "monarch", "orange",
        "anemone fish",  # clownfish - orange
        "admiral",  # red admiral butterfly
        "lorikeet",  # rainbow-colored
        "macaw",  # vivid reds/blues/yellows
        "flamingo",  # pink
        "indigo bunting",  # deep blue
        "peacock",  # iridescent blues/greens
        # Whites
        "white wolf", "white stork",
        "great pyrenees", "samoyed", "west highland white terrier",
        "ice bear",  # polar bear
        "arctic fox", "american egret",
        # Blacks
        "black swan", "black stork", "black widow",
        "black grouse", "american black bear", "black-footed ferret",
        "schipperke",
        # Browns
        "brown bear",
        # Greys
        "grey whale", "grey fox", "african grey",
        "great grey owl",
        # Blue
        "little blue heron", "kerry blue terrier",
        # Specific color names in class
        "sorrel",  # reddish-brown horse
    ]),

    ("Numbers & Numerals", [
        "the number ", "number 0", "number 1", "number 2", "number 3",
        "number 4", "number 5", "number 6", "number 7", "number 8",
        "number 9", "number 10", "numeral", "numbers in it",
        "mathematical formula", "the number eight", "the number five",
        "the number four", "the number nine", "the number seven",
        "the number six", "the number thirty", "the number three",
        "the number twelve", "the number twenty", "the number two",
        "the number fifteen", "the number eleven", "the number fourteen",
        "the number twenty-five",
        # ImageNet
        "scoreboard", "odometer", "abacus", "slide rule",
    ]),

    ("Letters, Text & Writing", [
        "the letter a", "the letter b", "the letter c", "the letter d",
        "the letter e", "the letter f", "the letter g", "the letter h",
        "the letter i", "the letter j", "the letter k", "the letter l",
        "the letter m", "the letter n", "the letter o", "the letter p",
        "the letter q", "the letter r", "the letter s", "the letter t",
        "the letter u", "the letter v", "the letter w", "the letter x",
        "the letter y", "the letter z",
        "english letters", "roman numeral", "calligraphy",
        "morse code", "handwritten", "italic text", "italic word",
        "bold text", "bold word", "short text", "long text",
        "graffiti with a sentence", "text in it", "vintage typography",
        "typography", "with calligraphy", "arabic script",
        "islamic calligraphy", "image with handwritten text",
        "image with calligraphy writing", "illustration with english letters",
        "illustration with roman numerals",
        # ImageNet
        "street sign", "book jacket", "menu", "comic book",
        "crossword puzzle", "web site",
    ]),

    ("Photography Techniques & Styles", [
        "bokeh", "long exposure", "motion blur", "wide angle", "wide-angle",
        "fisheye", "tilt-shift", "time-lapse", "slow shutter", "stop motion",
        "light painting", "light trails", "double exposure", "lens flare",
        "rule of thirds", "depth of field", "hdr", "high dynamic range",
        "motion freeze", "time-lapse image", "time-lapse trails",
        "time-lapse effect", "a blurry image", "a noisy photo",
        "high-resolution image", "low-resolution image",
        "artistic style of", "photograph with the artistic style of",
        "photo technique", "miniature diorama", "candid documentary",
        "candid portrait photography", "cinematic framing",
        "cinematic portrait", "dramatic chiaroscuro photography",
        "a zoomed in photo", "a zoomed out photo",
        "a photo taken at twilight", "a close-up shot",
        "an object centric photo", "action shot",
        # ImageNet
        "polaroid camera", "reflex camera", "tripod",
        "lens cap",
    ]),

    ("Lighting & Light Effects", [
        "lighting", "illuminat", "golden hour", " sunrise", " sunset",
        " dusk", " dawn", "twilight", "backlight", "high-key light",
        "low-key light", "artificial lighting", "natural lighting",
        "dramatic light", "play of light", "glowing", "luminous",
        "sunlit", "daytime illumin", "nighttime illumin",
        "neon light", "light and shadow", "shadow play",
        "light effect", "light show", "intentional lens flare",
        "glimmering light", "daytime scene", "daytime shot",
        "nighttime scene", "nighttime shot", "high contrast lighting",
        "high-key contrast", "low contrast", "moody lighting",
        "bokeh light", "glimmering light", "dappled sunlight",
        "play of shadows", "strong backlighting",
        # ImageNet
        "spotlight", "torch", "candle", "beacon",
        "lampshade", "table lamp", "matchstick",
    ]),

    ("Composition & Framing", [
        "balanced composition", "asymmetrical", "symmetrical composition",
        "rule of thirds", "leading lines", "negative space",
        "diagonal composition", "central focal point", "framing element",
        "frame within a frame", "bold composition", "dynamic composition",
        "wide-angle perspective", "panoramic view",
        "low-angle perspective", "unusual angle",
        "point of view from above", "point of view from below",
        "birds-eye view", "aerial perspective",
        "overlapping element", "multilayered depth",
        "spatial depth", "isolated subject", "lone subject",
        "balanced asymmetry", "focused subject",
        "strong leading lines", "dynamic leading lines",
        "diagonal composition", "visual rhythm",
    ]),

    ("Reflections & Mirrors", [
        "reflection in water", "reflection in mirror",
        " reflected", "reflective", " mirror ", "broken mirror",
        "shattered mirror", "mirror effect", "reflecting",
        "a reflection", "muted reflection",
        "abstract reflection", "city lights reflected",
        "reflection or mirror", "reflections on water",
        "reflective and calm lake", "reflective surface",
        "reflective modern glass",
        # ImageNet
        "car mirror",
    ]),

    ("Silhouettes & Shadows", [
        "silhouette", "shadow play",
        "dramatic shadows", "dynamic shadows",
        "dark silhouette", "play of shadows",
        "captivating silhouettes", "enigmatic silhouettes",
        "evocative silhouettes", "a shadow",
    ]),

    ("Black & White / Monochrome", [
        "black and white", "monochromatic", "grayscale", "monochrome",
        "sepia-tone", "sepia tone", "timeless black and white",
        "grayscale urban", "monotone", "achromatic",
        "crisp, monochrome", "high contrast black and white",
        "classic black and white", "portraits in black and white",
        "black and white vintage", "black and white candid",
        "contemplative monochrome", "timeless black",
        # ── ImageNet classes with strong B&W visual identity ──
        "dalmatian", "zebra", "giant panda", "skunk",
        "killer whale", "magpie", "black and gold garden spider",
        "black stork",  # black & white plumage
        "king penguin",  # black & white
        "panda",
    ]),

    ("Vintage & Retro Aesthetics", [
        "vintage", "retro", "nostalgic", "an old photo", "aged look",
        "sepia-toned photograph", "historical photograph",
        "old-world charm", "vintage filter", "vintage style",
        "retro style", "nostalgic mood", "nostalgic charm",
        "vintage nostalgia", "retro-style", "film grain effect",
        "old film effect", "vintage film", "washed-out vintage",
        "faded, nostalgic", "a vintage", "antique timepiece",
        "antique photo",
        # ImageNet
        "model t",  # vintage car
        "dial telephone", "typewriter keyboard",
        "cassette", "cassette player",
        "steam locomotive", "jinrikisha",
        "horse cart", "oxcart",
    ]),

    ("Macro & Close-up Photography", [
        "macro", "close-up", "close up",
        "a zoomed in photo", "macro shot", "macro photo",
        "macro detail", "macro botanical", "macro floral",
        "macro focus", "intimate close", "intense macro",
        "nature macro", "detailed macro",
        "captivating macro floral", "detailed amphibian close",
        "detailed animal close", "detailed arachnid close",
        "detailed botanical macro", "detailed insect close",
        "detailed insect macro", "detailed reptile close",
        "close-up view", "close-up of",
        # ImageNet (often photographed macro)
        "loupe",
    ]),

    ("Abstract Art & Patterns", [
        "abstract acrylic painting", "abstract artwork",
        "abstract composition", "abstract expressionist",
        "abstract geometric", "abstract form",
        "abstract oil painting", "abstract fractal",
        "abstract wave", "abstract patterns", "abstract graffiti",
        "abstract geometric shapes", "abstract geometric patterns",
        "abstract reflections", "conceptual abstraction",
        "conceptual exploration", "conceptual representation",
        "contemporary abstract painting",
    ]),

    ("Optical Illusions & Visual Effects", [
        "optical illusion", "illusion design",
        "distorted perspective", "illusion effect",
        "optical illusion artwork", "optical illusion design",
        "an optical illusion", "shattered reality",
        "double exposure effect", "image with a double exposure",
        # ImageNet
        "maze", "jigsaw puzzle",
    ]),

    # ══════════════════════════════════════════════════════════════════════════
    # SHAPES
    # ══════════════════════════════════════════════════════════════════════════

    ("Circles & Round Objects", [
        "a circle", "circular object", "concentric circle",
        "a semicircle", "a semi-circle", "a ring",
        "a hoop", "a sphere", "an oval", "an ellipse",
        "a bubble", "a disc", "a puck", "a wheel",
        "a globe", "a ball", "circular", "a tire",
        "a concentric", "round shape",
        # ImageNet
        "basketball", "baseball", "tennis ball", "golf ball",
        "soccer ball", "volleyball", "croquet ball", "ping-pong ball",
        "rugby ball",
        "bubble", "car wheel", "analog clock", "wall clock",
        "digital clock", "manhole cover",
        "plate", "goblet",
    ]),

    ("Triangles & Cones", [
        "a triangle", "triangular object", "equilateral triangle",
        "a right triangle", "a scalene triangle", "an acute triangle",
        "an obtuse triangle", "an isosceles triangle", "an inverted triangle",
        "a pyramid", "a cone", "a prism", "triangular",
        # ImageNet
        "stupa",  # conical/dome shape
        "teepee",
    ]),

    ("Squares, Rectangles & Grids", [
        "a square", "a rectangle", "rectangular object",
        "a quadrilateral", "a parallelogram", "a rhombus",
        "a trapezoid", "a right trapezoid", "a checkerboard",
        "checker pattern", "a grid", "grid-like",
        "scalene quadrilateral", "a decagon",
        # ImageNet
        "crossword puzzle", "jigsaw puzzle",
        "solar dish",
    ]),

    ("Stars & Star-shaped", [
        "a star", "a starburst", "a sunburst design",
        "a pentagram", "a hexagram", "a snowflake",
        "a fractal snowflake", "stardust", "a lightning bolt shape",
        # ImageNet
        "starfish",
    ]),

    ("Polygons & Multi-sided Shapes", [
        "a pentagon", "a hexagon", "a heptagon",
        "an octagon", "a decagon", "a dodecagon", "a polygon",
        "irregular polygon", "equilateral hexagon", "equilateral pentagon",
        "irregular pentagon", "irregular hexagon", "irregular heptagon",
        "irregular octagon", "regular octagon", "a kite",
        "a geometric tessellation", "geometric tessellation",
        "polygon with many sides",
    ]),

    ("Spirals, Swirls & Vortexes", [
        "a spiral", "a swirl", "a vortex", "a whirlpool",
        "a whirlwind", "a whirligig", "a helix",
        "a spirograph", "spiral pattern",
        "swirling", "a coil", "spiraling", "eddy",
        "swirling eddy", "swirling vortex", "a whirlpool",
        # ImageNet
        "chambered nautilus",  # spiral shell
        "coil",  # spiral coil
        "corkscrew",
        "conch",  # spiral shell
        "snail",  # spiral shell
    ]),

    ("Geometric Shapes (General)", [
        "geometric shape", "geometric pattern", "abstract geometric",
        "bold geometric", "geometric tessellation", "a diamond",
        "a curvilinear shape", "a freeform organic shape",
        "an irregular shape", "an oblong shape", "organic shape",
        "abstract form", "a teardrop shape", "a parabola",
        "a dodecagon", "geometric",
    ]),

    ("Patterns, Textures & Designs", [
        "pattern", "checkerboard", "honeycomb pattern",
        "polka dot", "zigzag pattern", "herringbone pattern",
        "paisley", "plaid pattern", "tartan pattern",
        "damask pattern", "chevron pattern", "argyle pattern",
        "striped design", "houndstooth", "tessellation",
        "mosaic arrangement", "woven", "ikat design",
        "floral pattern", "textile pattern", "fabric pattern",
        "camouflage pattern", "camouflage print", "leopard print",
        "zebra stripe pattern", "tie-dye pattern", "quilted pattern",
        "quilted fabric", "mandala", "celtic knotwork", "celtic spiral",
        "arabesque", "tribal pattern", "aboriginal dot",
        "aztec-inspired", "mayan-inspired", "greek key",
        "barcode", "crossword grid", "recurrent pattern",
        "ornate arabesque", "intricate mehndi", "gingham pattern",
        "harlequin pattern", "pointillism", "stippling technique",
        "paisley design", "a checkerboard", "a checker pattern",
        "a wavy pattern", "a wave pattern", "a polka dot",
        "a lattice design", "a mosaic", "a quilt pattern",
        "a woven fabric pattern", "a zebra stripe",
        "a honeycomb pattern", "a herringbone",
        "patchwork quilt", "patchwork design",
        "persian rug design", "plaid pattern", "quilted design",
        # ── ImageNet classes with visually distinctive patterns/textures ──
        # Striped animals
        "tiger", "tiger shark", "tiger cat", "tiger beetle",
        "zebra", "king snake", "garter snake", "coral snake",
        "ringneck snake", "banded gecko",
        # Spotted / dotted animals
        "dalmatian", "leopard", "snow leopard", "jaguar", "cheetah",
        "ladybug", "spotted salamander",
        # Textured shells & exoskeletons
        "armadillo",  # banded armor plates
        "pangolin",
        "trilobite",  # segmented texture
        "chiton",  # segmented shell plates
        "sea urchin",  # spiky texture
        "porcupine",  # quill texture
        "echidna",  # spiny texture
        # Web / lattice
        "spider web",
        "chainlink fence",  # lattice pattern
        "honeycomb",
        "chain mail",  # interlocking rings
        # Feather patterns
        "peacock",  # eye-spot pattern on feathers
        # Textured fabrics and materials
        "wool", "velvet", "quilt",
        "prayer rug",
        "doormat",
        # Textured surfaces
        "thatch",  # woven straw texture
        "worm fence",  # zigzag pattern
        "stone wall",  # stacked stone texture
        "tile roof",  # repeating tile pattern
        "picket fence",  # repeating slat pattern
        "knot",
    ]),

    ("Fractals & Complex Math", [
        "fractal", "fractal pattern", "fractal snowflake",
        "abstract fractal", "spiraling fractal", "fractal recursion",
        "intricate fractal", "a fractal", "artwork featuring abstract fractal",
        "artwork with fractal recursion", "artwork with spiraling fractal",
        "mesmerizing fractal",
        # ImageNet
        "brain coral",  # fractal-like branching surface
        "coral fungus",  # fractal branching
    ]),

    # ══════════════════════════════════════════════════════════════════════════
    # NATURE
    # ══════════════════════════════════════════════════════════════════════════

    ("Natural Landscapes", [
        "landscape", "a mountain", "mountains", "mountain range",
        "mountain vista", "mountain peak", "mountain landscape",
        "a meadow", "meadow landscape", "valley", "desert landscape",
        "glacier", "canyon", "tundra", "savanna", "rainforest",
        "grassland", "prairie", "a countryside", "rural scene",
        "wilderness", "a marsh", "pasture", "clearing", "woodland",
        "a vista", "open plains", "natural landscape",
        "natural wonder", "geological formation", "glade",
        "natural formation", "natural beauty", "fjord",
        "rolling vineyard", "rolling hills", "rolling countryside",
        "rolling wheat field", "wheat fields", "agricultural field",
        "farmland", "picturesque countryside",
        "serene countryside", "tranquil countryside",
        "peaceful countryside", "charming rural",
        "coastal landscape", "coastal view", "coastal lighthouse",
        "awe-inspiring mountain", "awe-inspiring natural",
        "majestic mountain", "awe-inspiring sky",
        "imposing mountain range",
        # ImageNet natural landforms
        "alp", "cliff", "valley", "promontory", "sandbar",
        "seashore", "lakeside",
        "rapeseed",  # fields of yellow
        "hay",  # agricultural landscape
    ]),

    ("Water & Aquatic", [
        "ocean", " sea ", "a lake", "a river", "a stream",
        "a waterfall", "underwater", "aquatic", "marine",
        " wave", "a puddle", "harbor", "a bay", "coast",
        "inlet", "cascade", "coral reef", " reef",
        "ripple", "waterscape", "waterside", "waterfront",
        "lakeside", "riverside", "oceanside", "seascape",
        "flowing water", "cascading waterfall", "a pond",
        "waterfall scene", "tranquil lake", "serene lake",
        "calm ocean", "ocean waves", "sea horizon",
        "ocean horizon", "open ocean", "a creek",
        "a meandering river", "a swirling eddy",
        "photo of a tranquil lake", "photo of a tranquil river",
        "picture with water", "flowing water bodies",
        "waterfall", "a droplet",
        # ImageNet water features
        "coral reef", "seashore", "sandbar", "lakeside",
        "breakwater", "dam", "dock", "pier",
        "fountain", "geyser",
    ]),

    ("Weather & Atmosphere", [
        "a cloud", "clouds", " rain", "a storm", "a tornado",
        "a rainbow", "thunder", "lightning storm",
        " fog", " mist", " haze", "snow", "a blizzard",
        "hurricane", "cyclone", "weather", "thunderstorm",
        "cloudy sky", "snowstorm", "sandstorm",
        "dramatic weather", "unpredictable weather",
        "gloomy weather", "rainy weather", "stormy weather",
        "a sandstorm", "foggy atmosphere", "misty environment",
        "image with a cloudy sky", "image with a dramatic thunderstorm",
        "image with a hurricane", "image with a lightning storm",
        "image with a snowstorm", "image with a tornado",
        "atmospheric haze", "dramatic clouds",
        # ImageNet
        "rain barrel", "umbrella", "snowplow",
        "barometer",
    ]),

    ("Sky & Space", [
        " sky", "space", " galaxy", "planet", "cosmic",
        "universe", "celestial", "aurora borealis",
        "constellation", "meteor", "quasar", "moon",
        "comet", "nebula", "starry night", "milky way",
        "astronomical", "cosmos", "interstellar", "intergalactic",
        "space exploration", "space station", "vast open sky",
        "sublime sky", "dramatic sky", "clear sky",
        "awe-inspiring sky", "enchanting starry",
        "captivating starry", "a quasar", "a satellite",
        "cloudless", "twinkling starlit sky",
        "cataclysmic skyline", "starlit sky",
        "a telescope", "image with stardust",
        "image with constellations", "image with cosmic energy",
        "a galaxy",
        # ImageNet
        "space shuttle", "radio telescope",
        "planetarium",
    ]),

    ("Plants & Botanical", [
        " plant", "a flower", " leaf", " grass", "botanical",
        "bloom", "petal", "a fern", "a cactus", "bamboo",
        " vine", "thistle", "lily", "magnolia", "tulip",
        "clover", "cloverleaf", "a palm", "pine tree",
        "a branch", "a stem", "blossom", "foliage",
        "flora", "a reed", "vegetation", "blossoming",
        "floral", "flowers", "a shrub",
        "picture of plants", "delicate flower petals",
        "macro botanical", "a petal",
        # ImageNet plants & flowers
        "daisy", "yellow lady's slipper",
        "rapeseed",  # bright yellow flowering plant
        "corn",  # crop plant
        "acorn",  # seed/nut
        "hip",  # rose hip
        "buckeye",  # nut/plant
        "head cabbage", "broccoli", "cauliflower",
        "artichoke", "cardoon",
        "bell pepper", "mushroom",
    ]),

    ("Trees & Forests", [
        " tree", "forest", "woodland", "jungle",
        "lush rainforest", "towering redwood", "a canopy",
        "image captured in a forest", "misty forest",
        "mysterious forest", "enchanting forest",
        "serene forest", "tranquil forest",
        "calming forest", "peaceful forest",
        "pristine forest", "magical forest",
        "forest clearing", "forest path", "forest scene",
        "forest glade", "forest glen", "forest view",
        "picture of trees", "picture taken in a forest",
        "tree branch", "a pine tree", "picture of trees",
        "towering redwood forest", "forest nymph",
        "forest haven", "forest refuge",
        # ImageNet
        "tree frog",  # lives in trees
        "lumbermill",
    ]),

    ("Fungi & Mushrooms", [
        # ── NEW GROUP for ImageNet mushroom/fungi classes ──
        "mushroom", "fungus", "fungi", "toadstool",
        "agaric", "bolete", "gyromitra", "stinkhorn",
        "earthstar", "hen-of-the-woods", "coral fungus",
    ]),

    # ══════════════════════════════════════════════════════════════════════════
    # URBAN
    # ══════════════════════════════════════════════════════════════════════════

    ("Urban & City Scenes", [
        "city life", "city lights", "city skyline",
        "city pulse", "city scene", "city square",
        "city traffic", "city waterfront", "city park",
        "city nightlife", "city intersection",
        "bustling city", "urban alleyway",
        "urban city", "urban street",
        "urban landscape", "urban setting", "urban reflection",
        "urban rhythm", "urban rooftop", "urban sanctuary",
        "urban spectacle", "urban street corner",
        "urban symphony", "urban vibrancy", "urban vitality",
        "urban solitude", "urban soul", "urban pulse",
        "urban perspectives", "urban mosaic", "urban life",
        "urban journeys", "urban intersection",
        "urban hustl", "urban explorations", "urban dreams",
        "urban dream", "urban diversity", "urban decay",
        "urban contrasts", "urban connections",
        "urban complexity", "urban cityscape",
        "urban authenticity", "urban architecture",
        "nostalgic city", "captivating city",
        "enigmatic city", "ethereal city",
        "evocative city", "subdued city",
        "whispering city", "bustling cityscape",
        "mysterious cityscape", "contemplative cityscape",
        "dynamic cityscape", "atmospheric cityscape",
        "atmospheric urban", "cityscape",
        # ImageNet urban
        "street sign", "traffic light", "parking meter",
        "manhole cover", "streetcar", "trolleybus",
        "pay-phone",
    ]),

    ("Architecture & Buildings", [
        "architectural", "architecture", "a building",
        "ancient structure", "a cathedral", "a church",
        "a temple", "a monument", "a tower",
        "a bridge", "a column", "an arch", "a dome",
        "a palace", "a castle", "a monastery",
        "architectural arches", "elegant victorian",
        "grand architecture", "ornate architectural",
        "ornate cathedral", "a lighthouse",
        "a staircase", "a window", "a roof",
        "majestic architecture", "majestic skyscrapers",
        "regal architecture", "innovative architectural",
        "futuristic architecture", "futuristic architectural",
        "modern skyscraper", "high-rise city architecture",
        "grand cathedral", "architectural detail",
        "architectural element", "architectural composition",
        "architectural contrast", "architectural elegance",
        "architectural expression", "architectural lines",
        "architectural marvel", "architectural reflection",
        "architectural revelation", "architectural rhythm",
        "architectural symmetry", "architectural symphony",
        "antique architectural", "ornate architectural",
        "majestic architectural", "weathered architecture",
        "awe-inspiring architectural",
        # ImageNet buildings & structures
        "church", "mosque", "monastery", "palace",
        "castle", "barn", "greenhouse", "prison",
        "library", "cinema", "restaurant",
        "bakery", "barbershop", "bookshop", "butcher shop",
        "confectionery", "grocery store", "shoe shop",
        "tobacco shop", "toyshop",
        "triumphal arch", "obelisk", "stupa",
        "beacon", "bell cote",
        "steel arch bridge", "suspension bridge", "viaduct",
        "dome", "vault",
        "water tower", "flagpole",
        "yurt", "cliff dwelling",
        "boathouse",
    ]),

    ("Industrial & Manufacturing", [
        "industrial", "factory", "machinery", "manufacturing",
        "warehouse", "construction site", "industrial landscape",
        "industrial backdrop", "industrial environment",
        "industrial factory machinery", "industrial construction",
        # ImageNet
        "lumbermill", "drilling platform",
        "forklift", "crane",
        "thresher", "harvester",
        "freight car", "container ship",
    ]),

    # ══════════════════════════════════════════════════════════════════════════
    # PEOPLE
    # ══════════════════════════════════════════════════════════════════════════

    ("People & Portraits", [
        "a portrait", "a photo of a man", "a photo of a woman",
        "a photo of an adult", "a photo of an old person",
        "a photo of a teenager", "a photo of a young person",
        "a picture of a baby", "a picture of a middle-aged person",
        "a picture of an elderly person", "photo of a person",
        "portrait of a person", "a group photo", "a family photo",
        "an image of a family", "an image of a couple",
        "an image of a body", "an image of a head",
        "an image of a face", "candid portrait",
        "artistic self-portrait", "reflective and introspective self-portrait",
        "an image of a king", "an image of a queen",
        "an image of three subjects", "an image of one subject",
        "an image of two subjects", "an image of friends hanging out",
        "an image capturing an interaction",
        "an image of a parent and child",
        "a photo of a man", "a photo of a woman",
        "a portrait", "portrait photography",
        "photo of a person", "photo of a woman",
        # ImageNet
        "ballplayer", "groom", "scuba diver",
    ]),

    ("Professions & Occupations", [
        "an image of a accountant", "an image of a actor",
        "an image of a aerospace engineer", "an image of a animal trainer",
        "an image of a arborist", "an image of a archaeologist",
        "an image of a architect", "an image of a art historian",
        "an image of a artist", "an image of a astronomer",
        "an image of a athlete", "an image of a attorney",
        "an image of a auto mechanic", "an image of a ballet dancer",
        "an image of a basketball player", "an image of a biologist",
        "an image of a carpenter", "an image of a chef",
        "an image of a chiropractor", "an image of a civil engineer",
        "an image of a composer", "an image of a dentist",
        "an image of a dermatologist", "an image of a detective",
        "an image of a doctor", "an image of a economist",
        "an image of a electrician", "an image of a emergency",
        "an image of a engineer", "an image of a farmer",
        "an image of a fashion designer", "an image of a film director",
        "an image of a financial analyst", "an image of a firefighter",
        "an image of a flight attendant", "an image of a florist",
        "an image of a gardener", "an image of a graphic designer",
        "an image of a gymnast", "an image of a hair stylist",
        "an image of a illustrator", "an image of a investment banker",
        "an image of a it specialist", "an image of a journalist",
        "an image of a judge", "an image of a landscaper",
        "an image of a lawyer", "an image of a librarian",
        "an image of a marine biologist", "an image of a mechanic",
        "an image of a mechanical engineer", "an image of a musician",
        "an image of a music producer", "an image of a news anchor",
        "an image of a novelist", "an image of a nurse",
        "an image of a orthopedic", "an image of a painter",
        "an image of a paramedic", "an image of a pediatrician",
        "an image of a pharmacist", "an image of a photographer",
        "an image of a pilot", "an image of a plumber",
        "an image of a podiatrist", "an image of a police",
        "an image of a preschool teacher", "an image of a private investigator",
        "an image of a professor", "an image of a psychologist",
        "an image of a radiologist", "an image of a scientist",
        "an image of a screenwriter", "an image of a social worker",
        "an image of a software developer", "an image of a surgeon",
        "an image of a swimmer", "an image of a systems analyst",
        "an image of a teacher", "an image of a veterinarian",
        "an image of a veterinary technician", "an image of a waiter",
        "an image of a welder", "an image of a writer",
        "an image of a zoologist",
        # ImageNet
        "military uniform", "lab coat",
    ]),

    ("Facial Expressions & Emotions", [
        "facial expression",
        "amused facial", "angry facial", "anxious facial",
        "astonished facial", "bewildered facial", "blissful facial",
        "bored facial", "cautious facial", "concerned facial",
        "confused facial", "content facial", "cynical facial",
        "despondent facial", "determined facial", "disappointed facial",
        "elated facial", "enthusiastic facial", "excited facial",
        "flustered facial", "focused facial", "giddy facial",
        "grumpy facial", "impatient facial", "indifferent facial",
        "inquisitive facial", "intense facial", "intrigued facial",
        "irritated facial", "joyful facial", "jubilant facial",
        "nonchalant facial", "nonplussed facial", "overwhelmed facial",
        "pensive facial", "playful facial", "radiant facial",
        "regretful facial", "relaxed facial", "relieved facial",
        "sad facial", "sarcastic facial", "serious facial",
        "shy facial", "skeptical facial", "smiling facial",
        "surprised facial", "thoughtful facial", "wistful facial",
        "winking facial", "affectionate smiling facial",
        "a happy feeling", "a sad feeling",
    ]),

    ("Body Parts", [
        "a hand", "an eye", "arms", " legs", "a nose", "a mouth",
        "cheeks", "ears", "fingers", "a head",
        "texture of hair", "texture of skin",
        "feet", "lips", "teeth", "thumb",
        "a paw", "a whisker", "a tail", "a fin",
        "a wing", "hands in an embrace",
        "image of cheeks", "image of ears",
        "image of legs", "image of hands",
        "eyes", "mouth",
        # ImageNet
        "ear",  # class 998
    ]),

    ("Children & Youth", [
        "a baby", "a toddler", "a teenager", "a young person",
        "a youngster", "children", "child", "youth",
        "adolescent", "infant", "endearing childhood",
        "playful children", "whimsical children",
        "energetic children", "joyful toddler",
        "innocent laughter", "cheerful adolescent",
        "energetic youngster", "enthusiastic youngster",
        "excited youth", "a picture of a baby",
        "playful siblings", "imaginative childhood",
        "whimsical childhood", "childhood fantasy",
        # ImageNet
        "cradle", "crib", "bassinet", "diaper", "bib",
    ]),

    ("Social Interactions & Connections", [
        "a family", "friends hanging out", "human connection",
        "an image of a couple", "an image of friends",
        "an image of a parent and child",
        "emotional and heartfelt connection",
        "emotional and heartfelt embrace",
        "emotional and heartfelt familial",
        "emotional and heartfelt family",
        "emotional and heartfelt friendship",
        "emotional and heartfelt human",
        "emotional candid embrace", "emotional candid interaction",
        "hands in an embrace", "candid interactions",
        "intimate and candid conversation",
        "intimate connection", "intimate moment",
        "photograph capturing friendship",
        "photograph expressing love",
        "playful interactions", "engaging interaction",
        "engaging dialogue", "heartwarming bonds",
        "family bonds",
    ]),

    # ══════════════════════════════════════════════════════════════════════════
    # ANIMALS (expanded + new sub-groups)
    # ══════════════════════════════════════════════════════════════════════════

    ("Animals", [
        "animal", "a cat ", "a dog ", "a wolf",
        "an elephant", "a zebra", "a sheep",
        "a penguin", " fish", "a snake", "a snail",
        "a scorpion", "mammal", "reptile", "amphibian",
        "rodent", "marsupial", "a seagull", " bird",
        "a hawk", "an owl", "insect", "a bee",
        "a butterfly", "a caterpillar", "a dragonfly",
        "a ladybug", "a spider", "wildlife", "feline",
        "a horse", "a cattle", "a donkey", "a cow",
        "a lion", "a tiger", "a bear", "a rabbit",
        "a duck", "a chicken", "a parrot",
        "a lizard", "a turtle", "a frog", "a whale",
        "a dolphin", "a shark", "a crab", "a seal",
        "a monkey", "a giraffe", "a camel",
        "a kangaroo", "a koala", "a raccoon",
        "a squirrel", "a fox", "with a bee",
        "with a butterfly", "with a caterpillar",
        "with a dragonfly", "with a hawk",
        "with a ladybug", "with an ant", "with an owl",
        "with a penguin", "with a spider",
        "with a seagull", "with a zebra", "with cats",
        "with dogs", "with seagulls", "with sheep",
        "with insects", "with poultry",
        "an image of fish", "image of a sheep",
        "photo of a reptile", "photo of a furry animal",
        "playful animals", "picture of animals",
        "picture of a feline", "majestic galloping horses",
        "graceful swimming fish", "graceful wings in motion",
        "majestic soaring birds", "curious wildlife",
        "candid wildlife", "dynamic wildlife",
        "playful zoo animal", "snapshot of a marsupial",
        "with birds", "a scorpion", "an ant",
        "an image with an ant", "image with a bee",
        "image with a butterfly", "image with a caterpillar",
        "image with a cattle", "image with a donkey",
        "image with a dragonfly", "image with dogs",
        "image with a hawk", "image with insects",
        "image with a ladybug", "image with an ant",
        "image with an owl", "image with a penguin",
        "image with a seagull", "image with a sheep",
        "image with a spider", "image with a zebra",
        "image with cats", "image with poultry",
        "image with seagulls", "image with sheep",
        "image showing prairie grouse",
        # ── ALL ImageNet animal classes ──
        # Fish
        "tench", "goldfish", "great white shark", "tiger shark",
        "hammerhead", "electric ray", "stingray",
        "barracouta", "eel", "coho", "rock beauty",
        "anemone fish", "sturgeon", "gar", "lionfish", "puffer",
        # Birds
        "cock", "hen", "ostrich", "brambling", "goldfinch",
        "house finch", "junco", "indigo bunting", "robin",
        "bulbul", "jay", "magpie", "chickadee", "water ouzel",
        "kite", "bald eagle", "vulture", "great grey owl",
        "black grouse", "ptarmigan", "ruffed grouse",
        "prairie chicken", "peacock", "quail", "partridge",
        "african grey", "macaw", "sulphur-crested cockatoo",
        "lorikeet", "coucal", "bee eater", "hornbill",
        "hummingbird", "jacamar", "toucan", "drake",
        "red-breasted merganser", "goose", "black swan",
        "white stork", "black stork", "spoonbill", "flamingo",
        "little blue heron", "american egret", "bittern",
        "limpkin", "european gallinule", "american coot",
        "bustard", "ruddy turnstone", "red-backed sandpiper",
        "redshank", "dowitcher", "oystercatcher", "pelican",
        "king penguin", "albatross",
        # Marine mammals
        "grey whale", "killer whale", "dugong", "sea lion",
        # Amphibians
        "european fire salamander", "common newt", "eft",
        "spotted salamander", "axolotl", "bullfrog",
        "tree frog", "tailed frog",
        # Reptiles
        "loggerhead", "leatherback turtle", "mud turtle",
        "terrapin", "box turtle", "banded gecko",
        "common iguana", "american chameleon", "whiptail",
        "agama", "frilled lizard", "alligator lizard",
        "gila monster", "green lizard", "african chameleon",
        "komodo dragon", "african crocodile", "american alligator",
        "triceratops",
        # Snakes
        "thunder snake", "ringneck snake", "hognose snake",
        "green snake", "king snake", "garter snake",
        "water snake", "vine snake", "night snake",
        "boa constrictor", "rock python", "indian cobra",
        "green mamba", "sea snake", "horned viper",
        "diamondback", "sidewinder",
        # Arachnids
        "trilobite", "harvestman", "scorpion",
        "black and gold garden spider", "barn spider",
        "garden spider", "black widow", "tarantula",
        "wolf spider", "tick", "centipede",
        # Insects
        "tiger beetle", "ladybug", "ground beetle",
        "long-horned beetle", "leaf beetle", "dung beetle",
        "rhinoceros beetle", "weevil", "fly", "bee", "ant",
        "grasshopper", "cricket", "walking stick",
        "cockroach", "mantis", "cicada", "leafhopper",
        "lacewing", "dragonfly", "damselfly",
        # Butterflies
        "admiral", "ringlet", "monarch",
        "cabbage butterfly", "sulphur butterfly", "lycaenid",
        # Marine invertebrates
        "starfish", "sea urchin", "sea cucumber",
        "jellyfish", "sea anemone", "brain coral",
        "flatworm", "nematode", "conch", "snail", "slug",
        "sea slug", "chiton", "chambered nautilus",
        # Crustaceans
        "dungeness crab", "rock crab", "fiddler crab",
        "king crab", "american lobster", "spiny lobster",
        "crayfish", "hermit crab", "isopod",
        # Mammals - monotremes & marsupials
        "tusker", "echidna", "platypus", "wallaby",
        "koala", "wombat",
        # Mammals - rodents & lagomorphs
        "wood rabbit", "hare", "angora", "hamster",
        "porcupine", "fox squirrel", "marmot", "beaver",
        "guinea pig",
        # Mammals - ungulates
        "sorrel", "zebra", "hog", "wild boar", "warthog",
        "hippopotamus", "ox", "water buffalo", "bison",
        "ram", "bighorn", "ibex", "hartebeest", "impala",
        "gazelle", "arabian camel", "llama",
        # Mammals - carnivores
        "weasel", "mink", "polecat", "black-footed ferret",
        "otter", "skunk", "badger", "armadillo",
        "three-toed sloth",
        # Primates
        "orangutan", "gorilla", "chimpanzee", "gibbon",
        "siamang", "guenon", "patas", "baboon", "macaque",
        "langur", "colobus", "proboscis monkey", "marmoset",
        "capuchin", "howler monkey", "titi", "spider monkey",
        "squirrel monkey", "madagascar cat", "indri",
        # Elephants
        "indian elephant", "african elephant",
        # Pandas
        "lesser panda", "giant panda",
        # Dogs (all breeds)
        "chihuahua", "japanese spaniel", "maltese dog",
        "pekinese", "shih-tzu", "blenheim spaniel",
        "papillon", "toy terrier", "rhodesian ridgeback",
        "afghan hound", "basset", "beagle", "bloodhound",
        "bluetick", "black-and-tan coonhound", "walker hound",
        "english foxhound", "redbone", "borzoi",
        "irish wolfhound", "italian greyhound", "whippet",
        "ibizan hound", "norwegian elkhound", "otterhound",
        "saluki", "scottish deerhound", "weimaraner",
        "staffordshire bullterrier",
        "american staffordshire terrier",
        "bedlington terrier", "border terrier",
        "kerry blue terrier", "irish terrier",
        "norfolk terrier", "norwich terrier",
        "yorkshire terrier", "wire-haired fox terrier",
        "lakeland terrier", "sealyham terrier", "airedale",
        "cairn", "australian terrier", "dandie dinmont",
        "boston bull", "miniature schnauzer", "giant schnauzer",
        "standard schnauzer", "scotch terrier",
        "tibetan terrier", "silky terrier",
        "soft-coated wheaten terrier",
        "west highland white terrier", "lhasa",
        "flat-coated retriever", "curly-coated retriever",
        "golden retriever", "labrador retriever",
        "chesapeake bay retriever",
        "german short-haired pointer", "vizsla",
        "english setter", "irish setter", "gordon setter",
        "brittany spaniel", "clumber", "english springer",
        "welsh springer spaniel", "cocker spaniel",
        "sussex spaniel", "irish water spaniel",
        "kuvasz", "schipperke", "groenendael", "malinois",
        "briard", "kelpie", "komondor",
        "old english sheepdog", "shetland sheepdog",
        "collie", "border collie", "bouvier des flandres",
        "rottweiler", "german shepherd", "doberman",
        "miniature pinscher", "greater swiss mountain dog",
        "bernese mountain dog", "appenzeller", "entlebucher",
        "boxer", "bull mastiff", "tibetan mastiff",
        "french bulldog", "great dane", "saint bernard",
        "eskimo dog", "malamute", "siberian husky",
        "dalmatian", "affenpinscher", "basenji", "pug",
        "leonberg", "newfoundland", "great pyrenees",
        "samoyed", "pomeranian", "chow", "keeshond",
        "brabancon griffon", "pembroke", "cardigan",
        "toy poodle", "miniature poodle", "standard poodle",
        "mexican hairless",
        # Cats & wild felines
        "tabby", "tiger cat", "persian cat", "siamese cat",
        "egyptian cat", "cougar", "lynx", "leopard",
        "snow leopard", "jaguar", "lion", "tiger",
        "cheetah",
        # Wild canines
        "timber wolf", "white wolf", "red wolf",
        "coyote", "dingo", "dhole", "african hunting dog",
        "hyena", "red fox", "kit fox", "arctic fox", "grey fox",
        # Bears
        "brown bear", "american black bear", "ice bear",
        "sloth bear",
        # Small mammals
        "mongoose", "meerkat",
    ]),

    ("Dogs & Canines", [
        # ── NEW GROUP: specific dog breed sub-group ──
        "a dog ", "with dogs", "image with dogs",
        # All ImageNet dog breeds
        "chihuahua", "japanese spaniel", "maltese dog",
        "pekinese", "shih-tzu", "blenheim spaniel",
        "papillon", "toy terrier", "rhodesian ridgeback",
        "afghan hound", "basset", "beagle", "bloodhound",
        "bluetick", "black-and-tan coonhound", "walker hound",
        "english foxhound", "redbone", "borzoi",
        "irish wolfhound", "italian greyhound", "whippet",
        "ibizan hound", "norwegian elkhound", "otterhound",
        "saluki", "scottish deerhound", "weimaraner",
        "staffordshire bullterrier",
        "american staffordshire terrier",
        "bedlington terrier", "border terrier",
        "kerry blue terrier", "irish terrier",
        "norfolk terrier", "norwich terrier",
        "yorkshire terrier", "wire-haired fox terrier",
        "lakeland terrier", "sealyham terrier", "airedale",
        "cairn", "australian terrier", "dandie dinmont",
        "boston bull", "miniature schnauzer", "giant schnauzer",
        "standard schnauzer", "scotch terrier",
        "tibetan terrier", "silky terrier",
        "soft-coated wheaten terrier",
        "west highland white terrier", "lhasa",
        "flat-coated retriever", "curly-coated retriever",
        "golden retriever", "labrador retriever",
        "chesapeake bay retriever",
        "german short-haired pointer", "vizsla",
        "english setter", "irish setter", "gordon setter",
        "brittany spaniel", "clumber", "english springer",
        "welsh springer spaniel", "cocker spaniel",
        "sussex spaniel", "irish water spaniel",
        "kuvasz", "schipperke", "groenendael", "malinois",
        "briard", "kelpie", "komondor",
        "old english sheepdog", "shetland sheepdog",
        "collie", "border collie", "bouvier des flandres",
        "rottweiler", "german shepherd", "doberman",
        "miniature pinscher", "greater swiss mountain dog",
        "bernese mountain dog", "appenzeller", "entlebucher",
        "boxer", "bull mastiff", "tibetan mastiff",
        "french bulldog", "great dane", "saint bernard",
        "eskimo dog", "malamute", "siberian husky",
        "dalmatian", "affenpinscher", "basenji", "pug",
        "leonberg", "newfoundland", "great pyrenees",
        "samoyed", "pomeranian", "chow", "keeshond",
        "brabancon griffon", "pembroke", "cardigan",
        "toy poodle", "miniature poodle", "standard poodle",
        "mexican hairless",
        # Wild canines
        "timber wolf", "white wolf", "red wolf",
        "coyote", "dingo", "dhole", "african hunting dog",
        "hyena", "red fox", "kit fox", "arctic fox", "grey fox",
        # Dog-related items
        "dogsled", "muzzle",
    ]),

    ("Cats & Felines", [
        # ── NEW GROUP: specific cat/feline sub-group ──
        "a cat ", "with cats", "image with cats",
        "picture of a feline", "feline",
        "tabby", "tiger cat", "persian cat", "siamese cat",
        "egyptian cat", "cougar", "lynx", "leopard",
        "snow leopard", "jaguar", "lion", "tiger",
        "cheetah", "madagascar cat",
    ]),

    ("Birds", [
        # ── NEW GROUP: specific bird sub-group ──
        " bird", "with birds", "majestic soaring birds",
        "graceful wings in motion", "with seagulls",
        "image with a seagull", "a seagull", "a hawk",
        "an owl", "a duck", "a chicken", "a parrot",
        "with a hawk", "with an owl", "with a penguin",
        "image with a penguin", "image with seagulls",
        "with poultry", "image with poultry",
        "image showing prairie grouse",
        # All ImageNet bird classes
        "cock", "hen", "ostrich", "brambling", "goldfinch",
        "house finch", "junco", "indigo bunting", "robin",
        "bulbul", "jay", "magpie", "chickadee", "water ouzel",
        "kite", "bald eagle", "vulture", "great grey owl",
        "black grouse", "ptarmigan", "ruffed grouse",
        "prairie chicken", "peacock", "quail", "partridge",
        "african grey", "macaw", "sulphur-crested cockatoo",
        "lorikeet", "coucal", "bee eater", "hornbill",
        "hummingbird", "jacamar", "toucan", "drake",
        "red-breasted merganser", "goose", "black swan",
        "white stork", "black stork", "spoonbill", "flamingo",
        "little blue heron", "american egret", "bittern",
        "limpkin", "european gallinule", "american coot",
        "bustard", "ruddy turnstone", "red-backed sandpiper",
        "redshank", "dowitcher", "oystercatcher", "pelican",
        "king penguin", "albatross",
        # Bird-related
        "birdhouse",
    ]),

    ("Reptiles & Amphibians", [
        # ── NEW GROUP ──
        "reptile", "amphibian", "a snake", "a lizard",
        "a turtle", "a frog",
        "photo of a reptile",
        "detailed reptile close", "detailed amphibian close",
        # Amphibians
        "european fire salamander", "common newt", "eft",
        "spotted salamander", "axolotl", "bullfrog",
        "tree frog", "tailed frog",
        # Turtles
        "loggerhead", "leatherback turtle", "mud turtle",
        "terrapin", "box turtle",
        # Lizards
        "banded gecko", "common iguana", "american chameleon",
        "whiptail", "agama", "frilled lizard",
        "alligator lizard", "gila monster", "green lizard",
        "african chameleon", "komodo dragon",
        # Crocodilians
        "african crocodile", "american alligator",
        "triceratops",
        # Snakes
        "thunder snake", "ringneck snake", "hognose snake",
        "green snake", "king snake", "garter snake",
        "water snake", "vine snake", "night snake",
        "boa constrictor", "rock python", "indian cobra",
        "green mamba", "sea snake", "horned viper",
        "diamondback", "sidewinder",
    ]),

    ("Insects & Arachnids", [
        # ── NEW GROUP ──
        "insect", "a bee", "a butterfly", "a caterpillar",
        "a dragonfly", "a ladybug", "a spider", "a scorpion",
        "an ant", "with a bee", "with a butterfly",
        "with a caterpillar", "with a dragonfly",
        "with a ladybug", "with an ant",
        "with a spider", "with insects",
        "image with a bee", "image with a butterfly",
        "image with a caterpillar", "image with a dragonfly",
        "image with a ladybug", "image with an ant",
        "image with insects", "image with a spider",
        "detailed insect close", "detailed insect macro",
        "detailed arachnid close",
        # ImageNet arachnids
        "trilobite", "harvestman", "scorpion",
        "black and gold garden spider", "barn spider",
        "garden spider", "black widow", "tarantula",
        "wolf spider", "tick", "centipede",
        # ImageNet insects - beetles
        "tiger beetle", "ladybug", "ground beetle",
        "long-horned beetle", "leaf beetle", "dung beetle",
        "rhinoceros beetle", "weevil",
        # Other insects
        "fly", "bee", "ant", "grasshopper", "cricket",
        "walking stick", "cockroach", "mantis", "cicada",
        "leafhopper", "lacewing", "dragonfly", "damselfly",
        # Butterflies
        "admiral", "ringlet", "monarch",
        "cabbage butterfly", "sulphur butterfly", "lycaenid",
        # Related
        "spider web", "apiary",  # beehive/bee yard
        "mosquito net",
    ]),

    ("Marine & Aquatic Life", [
        # ── NEW GROUP ──
        "underwater", "aquatic", "marine",
        "coral reef", " reef",
        "graceful swimming fish",
        "an image of fish",
        # ImageNet fish
        "tench", "goldfish", "great white shark", "tiger shark",
        "hammerhead", "electric ray", "stingray",
        "barracouta", "eel", "coho", "rock beauty",
        "anemone fish", "sturgeon", "gar", "lionfish", "puffer",
        # Marine mammals
        "grey whale", "killer whale", "dugong", "sea lion",
        # Marine invertebrates
        "jellyfish", "sea anemone", "brain coral",
        "flatworm", "nematode", "conch", "snail", "slug",
        "sea slug", "chiton", "chambered nautilus",
        "starfish", "sea urchin", "sea cucumber",
        # Crustaceans
        "dungeness crab", "rock crab", "fiddler crab",
        "king crab", "american lobster", "spiny lobster",
        "crayfish", "hermit crab", "isopod",
    ]),

    ("Primates", [
        # ── NEW GROUP ──
        "a monkey", "primate",
        "orangutan", "gorilla", "chimpanzee", "gibbon",
        "siamang", "guenon", "patas", "baboon", "macaque",
        "langur", "colobus", "proboscis monkey", "marmoset",
        "capuchin", "howler monkey", "titi", "spider monkey",
        "squirrel monkey", "madagascar cat", "indri",
    ]),

    # ══════════════════════════════════════════════════════════════════════════
    # FOOD & OBJECTS
    # ══════════════════════════════════════════════════════════════════════════

    ("Food & Cuisine", [
        "food", "fruit", "vegetable", "a dish", " meal",
        "beverage", "cuisine", "cooking", "a kitchen",
        "a bread", "a cake", "wine", "chocolate",
        "a cookie", "cheese", "pasta", "pizza",
        "a salad", "a soup", "dessert",
        "close up of food", "a photo of food",
        "an entree", "a main course", "a side dish",
        "fast food", "italian food",
        "close-up of a food item", "photo of food",
        "an image of a dish", "picture of fast food",
        "picture of italian food", "a fruit", "a vegetable",
        "a bowl", "a cup", "a plate", "a tea",
        "a cookie", "an image of a entree",
        "an image of a main course", "an image of a side dish",
        # ImageNet food items
        "guacamole", "consomme", "hot pot", "trifle",
        "ice cream", "ice lolly", "french loaf", "bagel",
        "pretzel", "cheeseburger", "hotdog", "mashed potato",
        "spaghetti squash", "acorn squash", "butternut squash",
        "carbonara", "chocolate sauce", "dough", "meat loaf",
        "pizza", "potpie", "burrito", "red wine",
        "espresso", "eggnog",
        "menu",  # restaurant menu
    ]),

    ("Fruits", [
        # ── NEW GROUP ──
        "a fruit", "fruit",
        # ImageNet specific fruits
        "granny smith", "strawberry", "orange", "lemon",
        "fig", "pineapple", "banana", "jackfruit",
        "custard apple", "pomegranate",
    ]),

    ("Vegetables & Produce", [
        # ── NEW GROUP ──
        "a vegetable", "vegetable", "produce",
        # ImageNet specific vegetables
        "head cabbage", "broccoli", "cauliflower",
        "zucchini", "spaghetti squash", "acorn squash",
        "butternut squash", "cucumber", "artichoke",
        "bell pepper", "cardoon", "mushroom",
        "corn", "rapeseed",
    ]),

    ("Sports & Athletics", [
        "sport", "athlete", "athletic", "competition",
        "a race", " match", "tournament", "marathon",
        "football", "soccer", "basketball", "swimming",
        "cycling", "running", "tennis", "golf",
        "baseball", "hockey", "volleyball", "skiing",
        "snowboarding", "surfing", "climbing", "gymnastics",
        "boxing", "wrestling", "motorsport", "racing event",
        "sports action", "sports challenge", "sports moment",
        "water sports", "extreme sports", "competitive sport",
        "sporting challenge", "an image of sports",
        "thrilling sports", "intense sports", "intense athlete",
        "intense athletic", "intense motorsport", "intense racing",
        "thrilling motorsport", "thrilling racing",
        "thrilling extreme sports", "focused athlete",
        "spirited sportsmanship", "powerful athletic",
        "dynamic sports", "intense extreme sports",
        # ImageNet sports items
        "baseball", "basketball", "tennis ball", "golf ball",
        "soccer ball", "volleyball", "rugby ball",
        "croquet ball", "ping-pong ball", "punching bag",
        "dumbbell", "barbell", "balance beam",
        "horizontal bar", "parallel bars",
        "football helmet", "knee pad", "ski", "ski mask",
        "bobsled", "scoreboard",
        "sports car", "racer", "racket",
        "running shoe", "swimming trunks",
        "ballplayer", "scuba diver",
    ]),

    ("Music, Dance & Performance", [
        "dance", "music concert", "a performance", "sing",
        "rhythm", "stage performance", "choir",
        "orchestra", "ballet", "recital",
        "dance performance", "dance routine", "dance competition",
        "music festival", "music performance",
        "live concert", "dance movement", "dance pose",
        "dynamic and energetic dance",
        "graceful ballet performance",
        "energetic and passionate music",
        "kinetic and lively dance",
        "evocative dance", "emotional dance",
        "dynamic and high-energy music",
        "dynamic and high-energy concert",
        "dynamic and high-energy dance",
        "pulsating concert",
        "a violin",
        # ImageNet - see also Musical Instruments group
        "stage", "theater curtain", "maypole",
    ]),

    ("Musical Instruments", [
        # ── NEW GROUP ──
        "a violin", "orchestra",
        # ImageNet instruments
        "accordion", "acoustic guitar", "electric guitar",
        "banjo", "bassoon", "cello", "chime",
        "cornet", "drum", "drumstick", "flute",
        "french horn", "gong", "harmonica", "harp",
        "maraca", "marimba", "oboe", "ocarina",
        "organ", "panpipe", "sax", "steel drum",
        "trombone", "upright",  # upright piano
        "grand piano", "violin",
    ]),

    ("Cultural & Traditional", [
        "cultural", "culture", "tradition", "ceremony", "ethnic",
        "heritage", "indigenous", "folklore", "ritual",
        "cultural festival", "cultural celebration",
        "cultural event", "cultural ceremony",
        "cultural performance", "cultural parade",
        "cultural market", "cultural exhibition",
        "vivid cultural", "colorful celebration",
        "colorful ceremony", "colorful festival",
        "colorful procession", "colorful event",
        "traditional cultural", "traditional festive",
        "vivid festive", "cultural mosaic",
        "cultural juxtaposition", "cultural tapestry",
        "cultural richness", "cultural stories",
        "cultural treasures", "time-honored tradition",
        # ImageNet
        "kimono", "abaya", "sarong", "sombrero",
        "prayer rug", "totem pole", "maypole",
        "altar",
    ]),

    ("Historical & Antique", [
        "ancient", "antique", "historical", "ruins", "artifact",
        "archaeological", "relic", "medieval",
        "ancient ruins", "ancient temple", "ancient castle",
        "ancient historical", "time-worn",
        "old-world", "historic cobblestone",
        "awe-inspiring ancient", "historical site",
        "crumbling ancient", "enduring historical",
        "timeless historical", "antique craftsmanship",
        "ancient civilization", "historical significance",
        "glimpse of the past", "whispers of history",
        "enduring cultural artifact",
        # ImageNet
        "triumphal arch", "obelisk", "megalith",
        "castle", "palace", "monastery",
        "sundial", "hourglass",
        "cannon", "guillotine",
        "breastplate", "cuirass", "chain mail",
    ]),

    ("Fantasy, Magic & Surreal", [
        "fantasy", "magical", "surreal", "enchanted",
        "mystical", "fairy-tale", "dreamlike", "dreamscape",
        "enchanting", "magical forest", "fantasy world",
        "fantasy realm", "mystical realm", "ethereal",
        "whimsical", "imaginative", "fantastical",
        "imagination", "dream world", "fairy tale",
        "storybook", "mythological", "fictional creature",
        "alien world", "otherworldly", "parallel universe",
        "illustration of a mystical", "illustration of a hidden fantasy",
        "illustration of a hidden enchanted",
        "illustration of a hidden celestial",
        "illustration of an enchanted",
        "illustration of an ethereal",
        "illustration of a dreamlike",
        "illustration of a utopian",
        "illustration of a mythical creature",
        "illustration of an alternate dimension",
        "illustration of a hidden ancient",
        "cosmic landscape", "enchanting  fantasy",
        "enchanting  mystical", "enchanting  twilight",
        "magical celestial", "magical fairy-tale",
        "magical dreamlike", "magical moonlit",
        "magical mystical",
        # ImageNet
        "triceratops",  # prehistoric/fantasy creature vibe
        "comic book",
        "jack-o'-lantern",
    ]),

    ("Transportation & Vehicles", [
        "a bicycle", "a bike", "a train", "an airplane",
        "a plane", "a boat", "a ship", "a helicopter",
        "a bus", "a truck", "a motorcycle", "a scooter",
        "a vehicle", "a car", "automobile", "a tractor",
        "an ambulance", "a police car", "a submarine",
        "a hot air balloon", "a rocket",
        "a horse-drawn carriage", "a skateboard",
        "a delivery van", "a garbage truck",
        "a construction vehicle", "image of a bicycle",
        "image of a boat", "image of a bus",
        "image of a car", "image of a helicopter",
        "image of a motorcycle", "image of an airplane",
        "image of a tractor", "image of a train",
        "image of a truck", "image with bikes",
        "image with boats", "picture with airplanes",
        "picture with cars", "picture with trains",
        "picture with boats", "advanced transportation",
        "futuristic transportation", "advanced transport system",
        "futuristic transport system", "image of a scooter",
        "image of a skateboard", "image of a submarine",
        "image of a rocket", "image of a ship",
        "a drone", "image of a policeman",
        "picture with boats",
        # ImageNet vehicles
        "aircraft carrier", "airliner", "airship",
        "ambulance", "amphibian",  # amphibious vehicle
        "beach wagon", "bicycle-built-for-two",
        "bobsled", "bullet train", "cab", "canoe",
        "car wheel", "catamaran", "convertible",
        "dogsled", "electric locomotive",
        "fire engine", "fireboat", "forklift",
        "freight car", "garbage truck", "go-kart",
        "golfcart", "gondola", "half track",
        "horse cart", "jeep", "jinrikisha",
        "lifeboat", "limousine", "liner",
        "minibus", "minivan", "model t", "moped",
        "motor scooter", "mountain bike", "moving van",
        "oxcart", "passenger car", "pickup",
        "police van", "racer", "recreational vehicle",
        "school bus", "schooner", "snowmobile",
        "snowplow", "space shuttle", "speedboat",
        "sports car", "steam locomotive", "streetcar",
        "submarine", "tow truck", "tractor",
        "trailer truck", "tricycle", "trimaran",
        "trolleybus", "unicycle", "warplane", "yawl",
        "container ship",
    ]),

    # ══════════════════════════════════════════════════════════════════════════
    # SPECIALIZED
    # ══════════════════════════════════════════════════════════════════════════

    ("Futuristic & Science Fiction", [
        "futuristic aesthetics", "futuristic architecture",
        "futuristic artificial intelligence",
        "futuristic biotechnology", "futuristic cityscapes",
        "futuristic design", "futuristic digital",
        "futuristic drone", "futuristic engineering",
        "futuristic innovations", "futuristic robotics",
        "futuristic robotic", "futuristic scientific",
        "futuristic skyline", "futuristic space exploration",
        "futuristic technological", "futuristic technology",
        "futuristic transportation", "futuristic transport",
        "futuristic-edge robotic", "futuristic-edge technology",
        "advanced artificial intelligence",
        "advanced biotechnology", "advanced drone technology",
        "advanced renewable energy", "advanced robotics",
        "advanced robotic technology", "advanced space exploration",
        "advanced transportation", "advanced transport system",
        "cutting-edge robotic", "cutting-edge technology",
        "innovative robotic technology",
        "innovative technological",
        "detailed illustration of a futuristic",
        "image with a futuristic",
        # ImageNet
        "space shuttle", "missile", "projectile",
    ]),

    ("Technology & Electronics", [
        "technology", "technological", "robot", "robotics",
        "artificial intelligence", "circuit board", "a circuit",
        "a circuit board", "a circuitry", "a sensor", "a capacitor",
        "a gear", "a motor", "a cable", "a wire", "a switch",
        "a satellite", "drone technology",
        "a keyboard", "a laptop", "a tablet", "a screen",
        "a camera", "a phone", "a joystick",
        "it specialist", "software developer", "systems analyst",
        # ImageNet electronics & tech
        "computer keyboard", "desktop computer", "laptop",
        "hand-held computer", "cellular telephone",
        "dial telephone", "pay-phone",
        "monitor", "screen", "television",
        "remote control", "joystick", "mouse",
        "modem", "hard disc", "cd player",
        "cassette player", "ipod", "tape player",
        "printer", "photocopier", "projector",
        "oscilloscope", "digital watch",
        "digital clock", "analog clock",
        "solar dish", "radio", "radio telescope",
        "microphone", "loudspeaker",
        "web site",
    ]),

    ("Fashion & Clothing", [
        "fashion", "clothing", "a dress", "a shirt", "a jacket",
        "a coat", "a skirt", "a suit", "outfit", "a boot",
        "a hat", "a scarf", "a tie", "a glove", "a sock",
        "garment", "fashion portrait", "fashion shot",
        "fashion silhouette", "fashion statement",
        "fashion show", "fashion runway",
        "striking fashion", "fashion attire",
        "urban street fashion", "eccentric fashion shot",
        "a turban", "a bonnet", "a belt", "a wallet",
        "a watch", "a mask",
        # ImageNet clothing & fashion
        "abaya", "academic gown", "apron", "bikini",
        "bib", "bonnet", "bolo tie", "bow tie",
        "brassiere", "cardigan", "cloak", "clog",
        "cowboy boot", "cowboy hat", "crash helmet",
        "feather boa", "fur coat", "gown", "hair slide",
        "hoopskirt", "jean", "jersey", "kimono",
        "knee pad", "lab coat", "loafer", "maillot",
        "military uniform", "miniskirt", "mitten",
        "mortarboard", "overskirt", "pajama",
        "poncho", "running shoe", "sandal", "sarong",
        "ski mask", "sock", "sombrero", "stole",
        "suit", "sunglasses", "sunglass",
        "sweatshirt", "swimming trunks", "trench coat",
        "vestment", "wig", "windsor tie",
    ]),

    ("Footwear", [
        # ── NEW GROUP ──
        "a boot", "a shoe",
        # ImageNet
        "cowboy boot", "clog", "loafer", "running shoe",
        "sandal",
    ]),

    ("Headwear & Headgear", [
        # ── NEW GROUP ──
        "a hat", "a turban", "a bonnet", "helmet",
        # ImageNet
        "bonnet", "cowboy hat", "crash helmet",
        "football helmet", "mortarboard",
        "pickelhaube", "shower cap", "ski mask",
        "sombrero", "bathing cap", "bearskin",
        "wig",
    ]),

    ("Jewelry & Accessories", [
        "jewelry", "a gemstone", "a necklace", "a bracelet",
        "an earring", "a pendant", "a crown", "a pearl",
        "a badge", "a gem", "a ring", "a jewel",
        "intricate jewelry design", "intricate gemstone",
        "a diamond", "ornament",
        # ImageNet
        "necklace", "buckle", "safety pin",
        "hair slide",
    ]),

    ("Medical & Scientific", [
        "medical procedure", "surgical procedure",
        "medical equipment", "scientific equipment",
        "precise medical", "precise surgical",
        "meticulous medical", "meticulous surgical",
        "precise scientific", "medical technology",
        "futuristic medical", "futuristic brain-computer",
        "futuristic nanotechnology", "futuristic quantum",
        "futuristic bioreactor", "futuristic biotechnology",
        "detailed illustration of a futuristic medical",
        "advanced medical technology",
        # ImageNet medical & scientific
        "stethoscope", "syringe", "pill bottle",
        "medicine chest", "neck brace", "crutch",
        "stretcher", "oxygen mask", "face powder",
        "petri dish", "beaker",
    ]),

    ("Graffiti & Street Art", [
        "graffiti", "street art", "a mural", "urban art",
        "bold graffiti", "street expression", "urban mural",
        "colorful graffiti", "abstract graffiti", "graffiti art",
        "artwork featuring graffiti",
        "street art-inspired", "urban and expressive",
        "urban and vibrant street art",
        "urban and gritty street art",
        "colorful urban art",
        "quirky street art",
    ]),

    ("Religious & Spiritual", [
        "a cathedral", "a temple", "spiritual",
        "religious", "sacred", "holy", "monastery",
        "religious icon", "antique religious icon",
        "tranquil asian temple", "ornate cathedral",
        "grand cathedral interior", "a cross",
        "a dreamcatcher", "islamic calligraphy",
        "ancient temple ruins", "tranquil temple courtyard",
        # ImageNet
        "church", "mosque", "monastery",
        "altar", "prayer rug", "stupa",
        "vestment",
    ]),

    ("Caricatures & Illustrations", [
        "caricature of",
        "a caricature",
        "detailed illustration of",
        "illustration of",
        "a drawing", "an illustration",
        "a painting", "detailed charcoal sketch",
        "intricate pencil drawing",
        "art nouveau-inspired",
        # ImageNet
        "comic book",
    ]),

    ("Artistic Styles & Movements", [
        "impressionist", "cubist", "surrealist art",
        "anime style image", "cartoon style image",
        "pop art", "expressionist artwork",
        "vibrant watercolor painting", "abstract oil painting",
        "abstract acrylic painting",
        "detailed charcoal sketch",
        "art nouveau", "pointillism technique",
        "8-bit pixel art", "glitch art aesthetic",
        "stained glass design", "chiaroscuro",
        "photograph with the artistic style",
        "artwork featuring retro tv test",
        "artwork featuring 8-bit",
        "artwork featuring abstract",
        "artwork featuring barcode",
        "artwork featuring circuit board",
        "artwork featuring crossword",
        "artwork featuring cubist",
        "artwork featuring digital glitch",
        "artwork featuring escher",
        "artwork featuring geometric tessellation",
        "artwork featuring graffiti",
        "artwork featuring herringbone",
        "artwork featuring labyrinthine",
        "artwork featuring morse code",
        "artwork featuring overlapping",
        "artwork featuring retro",
        "artwork featuring shattered glass",
        "artwork featuring typographic",
        "artwork featuring zebra stripe",
        "artwork with abstract fractal",
        "artwork with chaotic abstract",
        "artwork with fractal recursion",
        "artwork with glitch art",
        "artwork with intricate filigree",
        "artwork with kaleidoscopic",
        "artwork with mondrian",
        "artwork with mosaic",
        "artwork with optical illusion",
        "artwork with pixelated",
        "artwork with pointillism",
        "artwork with retro pixel",
        "artwork with retro video game",
        "artwork with spiraling fractal",
        "artwork with stained glass",
        "artwork with stippling",
        "artwork with woven basket",
        "classic artistic masterpiece",
        "timeless artistic masterpiece",
        "timeless fine art piece",
        "classic fine art piece",
        "exquisite fine art painting",
        "enduring classic artwork",
        "impressionist landscape painting",
        "impressionist portrait painting",
        "impressionist-style digital",
        "impressionist style",
        "cubist still life painting",
        "cubist composition",
        "contemporary abstract painting",
        "surreal artwork with",
        "surrealist artwork with",
        "surreal photo manipulation",
        "surreal digital collage",
        "surrealist collage artwork",
        "retro-style poster design",
        "anime style", "cartoon style",
    ]),

    ("Minimalism", [
        "minimalist", "minimal color palette",
        "minimalism", "stark minimalism",
        "minimalist composition", "minimalist lines",
        "minimalist urban", "minimalist architectural",
        "minimalist white backdrop",
        "minimalist design", "muted elegance",
        "quiet simplicity", "stark and minimalist",
    ]),

    ("Seasons", [
        "a photo taken in the fall", "a photo taken in the spring",
        "a photo taken in the summer", "a photo taken in the winter",
        "photograph taken during autumn",
        "photograph taken during spring",
        "photograph taken during winter",
        "crisp autumn leaves", "blossoming springtime",
        "autumn leaves", "vibrant autumn foliage",
        "pristine snowy landscape", "frozen wilderness",
        "snow-covered mountain peaks", "snowy forest trail",
    ]),

    ("Night & Darkness", [
        "night", "nighttime", "nocturnal", "moonlit",
        "city lights", "neon sign", "darkness",
        "night scene", "night setting", "night sky",
        "bustling city night", "nighttime shot",
        "nighttime scene", "atmospheric night",
        "mysterious night", "bustling city nightlife",
        "low-light condition", "fast-paced urban nightlife",
        "enchanting moonlit", "magical moonlit",
        "mystical moonlit", "mystical moonlit scene",
        # ImageNet
        "night snake",  # nocturnal
    ]),

    ("Celebrations & Festivals", [
        "festival", "celebration", "carnival",
        "fireworks", "a parade", "festive",
        "lively festival", "colorful festival",
        "vibrant festival", "cultural festival",
        "dynamic festival", "festive celebration",
        "joyful celebration", "carnival scene",
        "masquerade ball", "amusement park",
        "fairground", "vivid festival",
        "photo of a fireworks display", "whirling carousel",
        "whirling amusement park ride",
        "bursting fireworks display",
        "lively and colorful parade",
        "lively city parade", "lively carnival",
        "colorful hot air balloons",
        # ImageNet
        "carousel", "christmas stocking",
        "jack-o'-lantern", "maypole",
        "pinwheel", "balloon",
        "confetti",
    ]),

    ("Markets & Commerce", [
        "marketplace", "a bazaar", "a shop", "a store",
        "vendor", "market stall", "food market",
        "street market", "food stall", "outdoor market",
        "market life", "bustling market", "vibrant market",
        "busy market", "cultural market",
        "indian market", "spice market", "floating market",
        "market scene", "photo of a bustling marketplace",
        "image of street markets", "vibrant marketplace",
        "busy market square", "buzzing market square",
        "busy market square", "lively market scene",
        # ImageNet shops
        "bakery", "barbershop", "bookshop", "butcher shop",
        "confectionery", "grocery store", "shoe shop",
        "tobacco shop", "toyshop",
        "shopping basket", "shopping cart",
        "cash machine", "vending machine",
    ]),

    ("Locations & Places", [
        "image taken in", "photo taken in", "picture taken in",
        "image captured in", "image snapped in",
        "picture captured in", "picture snapped in",
        "photo captured in",
        "an image of andorra", "an image of barcelona",
        "an image of dublin", "an image of fiji",
        "an image of glasgow", "an image of kenya",
        "an image of liechtenstein", "an image of luxembourg",
        "an image of monaco", "an image of namibia",
        "an image of portsmouth", "an image of samoa",
        "a photo of cardiff", "a photo of glasgow",
        "a photo of illinois", "a photo of manchester",
        "a photo of monaco", "a picture of south korea",
        "a picture of taiwan", "a picture of wisconsin",
        "a picture of liechtenstein", "a picture of samoa",
        "a picture of illinois", "image snapped in spain",
        "a photo of serene countryside",
        "image snapped in the",
    ]),

    ("Aerial & Bird's-Eye Views", [
        "aerial view", "aerial perspective",
        "aerial landscape photography",
        "point of view from above",
        "aerial landscape", "drone view",
        "bustling city from above",
        "photo taken from above",
        "birds-eye view",
        # ImageNet
        "parachute",  # often aerial view
        "airship",  # aerial perspective
    ]),

    ("Symmetry & Balance", [
        "symmetr", "symmetrical", "radial symmetry",
        "harmonic symmetry", "play of symmetry",
        "dynamic symmetry", "natural symmetry",
        "horizontal symmetry", "vertical symmetry",
        "architectural symmetry", "asymmetrical arrangement",
        "balanced composition", "balanced asymmetry",
        # ImageNet (visually symmetric)
        "butterfly",  # bilateral symmetry
        "starfish",  # radial symmetry
        "pinwheel",
    ]),

    ("Mood & Atmosphere (Serene / Peaceful)", [
        "serene", "tranquil", "peaceful", "calming",
        "soothing", "restful", "placid", "idyllic",
        "quiet simplicity", "quiet solitude", "quiet and serene",
        "serene beach", "tranquil boating", "tranquil landscapes",
        "serene landscape", "serene waterscape",
        "peaceful lakeside", "serene lakeside",
        "serene garden", "tranquil garden",
        # ImageNet
        "lakeside",
    ]),

    ("Mood & Atmosphere (Dramatic / Mysterious)", [
        "mysterious atmosphere", "mysterious ambiance",
        "enigmatic atmosphere", "enigmatic ambiance",
        "dramatic", "haunting", "atmospheric mood",
        "atmospheric cityscape", "atmospheric twilight",
        "atmospheric urban", "atmospheric haze",
        "subdued atmosphere", "moody", "eerie",
        "dark and moody", "gritty realism",
        "gritty urban", "film noir",
        "ominous", "tense",
        # ImageNet
        "volcano",  # dramatic natural scene
        "wreck",  # dramatic scene
    ]),

    ("Subject Count", [
        "single subject", "one subject", "two subjects",
        "three subjects", "a pair of subjects", "a duo of",
        "a trio of", "a couple of subjects",
        "with a four people", "with a five people",
        "with a six people", "with a seven people",
        "with a group of", "a crowd of",
        "multiple subjects", "many subjects",
        "numerous subjects", "a bunch of subjects",
        "a handful of subjects", "a cluster of subjects",
        "a team of subjects", "a range of subjects",
        "a selection of subjects", "a set of subjects",
        "image with a single subject", "image with two subjects",
        "image with three subjects",
        "image with a couple of subjects",
        "image with a crowd of subjects",
        "image with a group of subjects",
        "image with a pair of subjects",
        "image with a duo of",
        "image with a trio of",
        "image with multiple subjects",
        "image with a bunch of subjects",
        "image with a handful of subjects",
        "image with a cluster of subjects",
        "image with a team of",
        "image with a range of subjects",
        "image with a selection of subjects",
        "image with a set of subjects",
        "image with five subjects",
        "image with six subjects",
        "image with three people",
        "image with many subjects",
        "image with numerous subjects",
        "image with several subjects",
        "image with two subjects",
        "an image of one subject",
        "an image of three subjects",
        "an image of two subjects",
    ]),

    ("Fire, Smoke & Volcanic", [
        "fire", "a lava", "volcanic eruption",
        "ember", "flame", "a volcano",
        "glowing embers", "dramatic volcanic eruption",
        "image with elemental magic and fire",
        "image with fire", "a swirl of smoke",
        "a smoky plume",
        # ImageNet
        "volcano", "torch", "matchstick",
        "fire engine", "fire screen", "fireboat",
    ]),

    ("Fog, Mist & Haze", [
        "foggy", "misty", "hazy", "atmospheric haze",
        "dreamy mist", "mystical fog", "morning mist",
        "fog", "mist", "a mist", "a fog",
        "swirling fog", "dreamy haze",
        "dreamlike haze", "dreamy misty morning",
    ]),

    ("Sunsets & Sunrises", [
        "a sunset", "a sunrise", "golden hour",
        "serene sunset", "dramatic sunset",
        "ocean sunset silhouette", "mountain peak sunrise",
        "serene sunrise or sunset",
        "picture taken at sunset or sunrise",
        "soothing beach sunset", "tranquil beach sunset",
        "serene beach sunset", "sunrise or sunset",
    ]),

    ("Timekeeping & Clocks", [
        "a clock", "a watch", "a pendulum",
        "intricate clock mechanism", "intricate clockwork gears",
        "intricate watch gears", "intricate watch mechanism",
        "intricate timekeeping mechanism", "precise clock mechanism",
        "precise clockwork gears", "precise watch gears",
        "precise watch mechanism", "precise timekeeping mechanism",
        "antique timepiece", "ornate timepiece",
        "timeless clock tower", "intricate pocket watch",
        "precise pocket watch",
        # ImageNet
        "analog clock", "digital clock", "wall clock",
        "digital watch", "hourglass", "sundial",
        "stopwatch",
    ]),

    ("Textures & Materials (Close-up)", [
        "close-up of a textured", "texture of",
        "close-up of textures", "natural texture",
        "collage of textures", "a smooth texture",
        "a spiky texture", "cracked surface texture",
        "fractured glass texture", "glass texture",
        "a marbled texture", "a houndstooth texture",
        "a quilted texture", "nature's textures",
        "captivating textures", "rich textures",
        "rustic wooden textures",
        # ImageNet materials with prominent texture
        "wool", "velvet", "thatch",
        "stone wall", "chain mail",
        "worm fence",
    ]),

    # ══════════════════════════════════════════════════════════════════════════
    # NEW GROUPS FOR IMAGENET OBJECTS
    # ══════════════════════════════════════════════════════════════════════════

    ("Household Objects & Furniture", [
        # ── NEW GROUP for ImageNet household items ──
        "bookcase", "chest", "chiffonier", "china cabinet",
        "cradle", "crib", "desk", "dining table",
        "entertainment center", "four-poster",
        "folding chair", "rocking chair",
        "park bench", "studio couch",
        "table lamp", "throne", "wardrobe",
        "bathtub", "shower curtain", "toilet seat",
        "washbasin", "doormat", "pillow",
        "window screen", "window shade",
        "vase", "flowerpot",
        "vacuum", "washer", "dishwasher",
        "iron", "sewing machine",
        "electric fan", "space heater",
        "refrigerator", "stove", "microwave",
        "toaster", "waffle iron",
        "lamp", "curtain", "bannister",
        "sliding door", "shoji",
    ]),

    ("Kitchen & Cooking Equipment", [
        # ── NEW GROUP ──
        "a kitchen", "cooking",
        "caldron", "coffeepot", "espresso maker",
        "crock pot", "dutch oven", "frying pan",
        "ladle", "measuring cup", "mixing bowl",
        "pot", "spatula", "strainer",
        "wok", "wooden spoon", "rotisserie",
        "can opener", "cleaver", "corkscrew",
        "cocktail shaker", "beer bottle", "beer glass",
        "coffee mug", "cup", "goblet",
        "pitcher", "soup bowl", "teapot",
        "plate rack", "tray", "saltshaker",
        "water bottle", "water jug", "wine bottle",
        "whiskey jug", "pop bottle",
    ]),

    ("Tools & Hardware", [
        # ── NEW GROUP ──
        "a gear", "a motor",
        "hammer", "hatchet", "screwdriver", "screw",
        "nail", "plunger", "power drill", "shovel",
        "carpenter's kit", "chain saw",
        "lawn mower", "plow", "broom",
        "paintbrush", "hook", "padlock",
        "combination lock", "rule",  # ruler
        "thimble",
    ]),

    ("Weapons & Armor", [
        # ── NEW GROUP ──
        "assault rifle", "rifle", "revolver",
        "cannon", "missile", "projectile",
        "bulletproof vest", "breastplate", "cuirass",
        "chain mail", "shield", "scabbard",
        "holster",
    ]),

    ("Containers & Vessels", [
        # ── NEW GROUP ──
        "barrel", "basket", "bucket", "carton", "crate",
        "hamper", "packet", "plastic bag",
        "rain barrel", "safe", "chest",
        "mail bag", "mailbox", "piggy bank",
        "purse", "wallet", "backpack",
        "pencil box", "medicine chest",
    ]),

    ("Toys, Games & Recreation", [
        # ── NEW GROUP ──
        "teddy",  # teddy bear
        "jigsaw puzzle", "crossword puzzle",
        "pinwheel", "swing", "carousel",
        "slot",  # slot machine
        "pool table", "maraca",
        "toyshop",
    ]),

    ("Optical & Scientific Instruments", [
        # ── NEW GROUP ──
        "binoculars", "loupe", "magnetic compass",
        "microscope", "oscilloscope",
        "projector", "radio telescope",
        "reflex camera", "polaroid camera",
        "stethoscope", "syringe", "petri dish",
        "beaker", "barometer",
    ]),

    ("Textiles & Fabrics", [
        # ── NEW GROUP ──
        "wool", "velvet", "quilt",
        "bath towel", "handkerchief", "dishrag",
        "paper towel", "binder",
        "sleeping bag",
        "prayer rug", "doormat",
        "shower curtain",
    ]),

    ("Structures & Barriers", [
        # ── NEW GROUP: fences, walls, bridges, dams ──
        "chainlink fence", "picket fence",
        "worm fence", "stone wall",
        "breakwater", "dam",
        "steel arch bridge", "suspension bridge",
        "viaduct", "pier", "dock",
        "turnstile", "manhole cover",
    ]),

    ("Outdoor & Camping", [
        # ── NEW GROUP ──
        "mountain tent", "sleeping bag",
        "backpack", "canoe", "paddle",
        "snorkel", "parachute",
        "recreational vehicle",
        "mobile home", "yurt",
    ]),

    ("Office & Stationery", [
        # ── NEW GROUP ──
        "envelope", "binder", "file",
        "pencil box", "pencil sharpener",
        "letter opener", "fountain pen",
        "ballpoint", "quill", "notebook",
        "typewriter keyboard", "desk",
        "rubber eraser",
    ]),

    ("Natural Landforms & Geology", [
        # ── NEW GROUP for ImageNet landscape classes ──
        "alp", "cliff", "coral reef", "geyser",
        "lakeside", "promontory", "sandbar",
        "seashore", "valley", "volcano",
        "cliff dwelling", "megalith",
    ]),
])

# ─── Assign items to groups ──────────────────────────────────────────────────
group_items = defaultdict(list)
unassigned = []

for source, item in all_items:
    item_norm = normalize(item)
    assigned_to_any = False
    for group_name, keywords in GROUPS.items():
        if matches(item_norm, keywords):
            group_items[group_name].append((source, item))
            assigned_to_any = True
    if not assigned_to_any:
        unassigned.append((source, item))

# ─── Statistics ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"CATEGORIZATION RESULTS")
print(f"{'='*70}")
print(f"Total items:        {len(all_items)}")
print(f"  Description texts: {len(texts)}")
print(f"  ImageNet classes:  {len(imagenet_classes)}")
print(f"Groups defined:     {len(GROUPS)}")
print(f"Unassigned items:   {len(unassigned)}")

# Count how many groups each item belongs to
membership_counts = defaultdict(int)
for group_name, items in group_items.items():
    for (src, item) in items:
        membership_counts[(src, item)] += 1

total_memberships = sum(membership_counts.values())
avg_membership = total_memberships / len(all_items) if all_items else 0
print(f"Avg groups/item:    {avg_membership:.2f}")

# Separate stats for texts vs imagenet
text_memberships = [v for (src, _), v in membership_counts.items() if src == "text"]
inet_memberships = [v for (src, _), v in membership_counts.items() if src == "imagenet"]
if text_memberships:
    print(f"Avg groups/text:    {sum(text_memberships)/len(texts):.2f}")
if inet_memberships:
    print(f"Avg groups/class:   {sum(inet_memberships)/len(imagenet_classes):.2f}")

print(f"\nGroup sizes (descending):")
for group, items in sorted(group_items.items(), key=lambda x: len(x[1]), reverse=True):
    n_text = sum(1 for s, _ in items if s == "text")
    n_inet = sum(1 for s, _ in items if s == "imagenet")
    print(f"  {len(items):5d}  (txt:{n_text:4d} inet:{n_inet:3d})  {group}")

# Unassigned breakdown
unassigned_texts = [(s, i) for s, i in unassigned if s == "text"]
unassigned_inet = [(s, i) for s, i in unassigned if s == "imagenet"]

if unassigned_texts:
    print(f"\nUnassigned texts ({len(unassigned_texts)}):")
    for _, t in unassigned_texts[:40]:
        print(f"  - {t}")
    if len(unassigned_texts) > 40:
        print(f"  ... and {len(unassigned_texts) - 40} more")

if unassigned_inet:
    print(f"\nUnassigned ImageNet classes ({len(unassigned_inet)}):")
    for _, c in unassigned_inet:
        print(f"  - {c}")

# ─── Save results ─────────────────────────────────────────────────────────────
output = {
    "metadata": {
        "total_items": len(all_items),
        "total_texts": len(texts),
        "total_imagenet_classes": len(imagenet_classes),
        "total_groups": len(GROUPS),
        "unassigned_count": len(unassigned),
        "unassigned_texts": len(unassigned_texts),
        "unassigned_imagenet": len(unassigned_inet),
        "average_groups_per_item": round(avg_membership, 2),
    },
    "groups": {},
    "unassigned": {
        "texts": sorted([i for _, i in unassigned_texts]),
        "imagenet": sorted([i for _, i in unassigned_inet]),
    },
}

for name, items in group_items.items():
    output["groups"][name] = {
        "texts": sorted(set(i for s, i in items if s == "text")),
        "imagenet_classes": sorted(set(i for s, i in items if s == "imagenet")),
    }

with open('/home/nfm/ViT-Prisma/mynotebooks/groups_output_combined.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved JSON to /home/nfm/ViT-Prisma/mynotebooks/groups_output_combined.json")
