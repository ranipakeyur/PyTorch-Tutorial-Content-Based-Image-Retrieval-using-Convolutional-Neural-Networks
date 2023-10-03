# PyTorch Tutorial: Content Based Image Retrieval using Convolutional Neural Networks

The complete article on this notebook is available on [jypynotes.com](https://www.jupynotes.com/python/pytorch-tutorial-content-based-image-retrieval-using-convolutional-neural-networks/).

Image retrieval is one of the foundational tasks in the realm of computer vision. When given a query image, it entails the search and identification of similar images within a vast dataset. Content-based image retrieval systems, at their core, utilize computer vision techniques to analyze textures, colors, shapes, and distinctive features of images. In this tutorial, we will employ a Convolutional Neural Network (CNN) to encode a d-dimensional feature descriptor of an image. While it is possible to use off-the-shelf (pre-trained) CNNs to extract feature descriptors, this may not be optimal for the target application. Therefore, we will train and fine-tune the model for the specific task to enhance its performance.

The complete source code and Colab notebook for this tutorial are available on my [GitHub repo](https://github.com/ranipakeyur/image_retrieval/blob/main/pytorch_tutorial_content_based_image_retrieval_v1.ipynb).


**Table of Contents:** <br>
1. [Dataset - CUB-200-2011](#Dataset-CUB-200-2011)
2. [pytorch_metric_learning and faiss](#2-pytorch_metric_learning-and-faiss)
3. [Configuration](#3-configuration) 
4. [Dataset and DataLoader](#4-dataset-and-dataloader)
5. [Image Retrieval Model](#5-image-retrieval-model)
6. [Training](#6-training)
7. [Evaluation - Quantitative and Qualitative Analysis](#7-evaluation)
8. [References](#8-references)


## 1. Dataset - CUB-200-2011 <a name="Dataset-CUB-200-2011"></a>


<div align="center" markdown="1">
<img src="/assets/images/post-0008/cub-200-2011-birds-dataset.png" alt="Example images from CUB-200-2011 dataset."/>
<figcaption align="center" markdown="1">
Figure 1. Example images from CUB-200-2011 dataset.
</figcaption>
</div>

<div align="center" markdown="1">
<img src="/assets/images/post-0008/sparrow_birds_cub_200-2011.png" alt="EExample of Sparrow Images."/>
<figcaption align="center" markdown="1">
Figure 2. Example of Sparrow Categories (Nearly Visually Indistinguishable) 
</figcaption>
</div>

In this tutorial, we will use **Caltech-UCSD Birds-200-2011 [(CUB-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/)** dataset, which is widely used as a benchmark for fine-grained visual categorization (of bird species) task. Bird species classification task is [challenging due to](https://authors.library.caltech.edu/records/cvm3y-5hh21/files/CUB_200_2011.pdf), 

> - Although different bird species share the same
basic set of parts, different bird species can vary dramati-
cally in shape and appearance (e.g., consider pelicans vs.
sparrows).
> - At the same time, other pairs of bird species
are nearly visually indistinguishable, even for expert bird
watchers (e.g. Fig 2., many sparrow species are visually similar).

For these reasons, this dataset would be an ideal challenge for an image retrieval task as well. This dataset contains 11,788 images
of 200 bird species. Each image has detailed annotations: 1 bird species class label, 15 part locations, 312 binary attributes and 1 bounding box. 

### Download and Extract Dataset

```shell
mkdir cub200
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz -P ./cub200
tar -xf ./cub200/CUB_200_2011.tgz -C ./cub200/
```

## 2. pytorch_metric_learning and faiss 
In this tutorial we will use [`pytorch_metric_learning`](https://github.com/KevinMusgrave/pytorch-metric-learning), which is a Python library for metric learning using PyTorch, offering pre-implemented loss functions, mining strategies, and evaluation metrics. It simplifies the implementation of similarity-based machine learning tasks, such as image retrieval and clustering, by providing ready-to-use tools and algorithms. In addition we will also use [`faiss`](https://github.com/facebookresearch/faiss), a fast and efficient library for similarity search and clustering of dense vectors, widely used in machine learning applications.

~~~shell
pip install pytorch-metric-learning
pip install faiss-gpu
~~~

## 3. Configuration

~~~python
config = {
    'dataset_root':'./cub200/CUB_200_2011/images/',
    'batch_size':128,
    'm_per_sample':4, # must be-> batch_size % m_per_sample = 0,
    'num_workers':2,
    'training': {
        'num_epochs':3,
        'smoothing':0.1,
        'margin':0.1,
        'temprature':0.5,
    },
    'model': {
        'backbone':'resnet50', # the only option as of now, pretrained from torchvision
        'feature_dim':1536,
        'num_classes':100,
    },
    'mode':'test', # train or test
    'model_save_path': './cub200/cub_checkpoint_image_MG_2.pth'
}
~~~

The dictionary above contains configurations for training and evaluating the image retrieval model.

### Training

For the training, use `mode='train'`. We will use [`ResNet50`](https://arxiv.org/abs/1512.03385) pre-trained on [ImageNet](https://www.image-net.org/) as the backbone network. By default, the `model_save_path` will save the model to the local storage in [Colab](https://colab.research.google.com/), which may be wiped out after runtime disconnects. Therefore, make sure to download the model before disconnecting the runtime or use your Google Drive to save model checkpoints.

### Evaluation

For evaluation, set `mode='test'`.

## 4. Dataset and DataLoader

The dataset contains 11,788 images. For our task, the dataset is split into train and test sets. These sets are class-disjoint. The train set has **5,864** images consisting of class IDs 1-100, and the test set has **5,924** images consisting of class IDs 101-200. In this tutorial, we are going to use cropped images for training the model. Each image in the dataset is annotated with a single bounding box. We will use these annotations to crop images. The following helper functions will be used to load bounding box annotations and then map them to corresponding images.

~~~python
def read_bbox(path):
    data = {}
    for line in open(path, 'r', encoding='utf-8'):
        id, l, t, w, h = line.split()
        data[id] = [l, t, w, h]
    return data

def read_image_paths(path):
    data = {}
    for line in open(path, 'r', encoding='utf-8'):
        id, name = line.split()
        data[id] = name
    return data

def map_image_name_bbox(image_folder, dataset_root):
    image_names = read_image_paths(os.path.join(dataset_root,'images.txt'))
    bboxes = read_bbox(os.path.join(dataset_root,'bounding_boxes.txt'))
    bbox_map = {}
    for img_id, img_name in image_names.items():
        l, t, w, h = bboxes[img_id]
        l_, t_ = int(float(l)), int(float(t))
        r_, b_ = l_ + int(float(w)), t_ + int(float(h))
        bbox_map[os.path.join(image_folder,img_name)] = (l_,t_,r_,b_)
    return bbox_map
~~~

### Class-Disjoint Train and Test Sets

Now, let us create a `ClassDisjointDataset` class. We will use the `ImageFolder` class from `torchvision` to create a dataset object. Then, we pass this object and the `image_bbox_map` to `ClassDisjointDataset`, which will split the dataset into class-disjoint train and test sets.

In the `__getitem__` function, you can notice that the image is cropped using bounding box annotations and then resized to the size of 256x256.


~~~python
# Define Dataset
image_bbox_map = map_image_name_bbox(os.path.join(DATA_ROOT,'images'),DATA_ROOT)

cub_set = datasets.ImageFolder(config['dataset_root'])

class ClassDisjointDataset(torch.utils.data.Dataset):
    def __init__(self, data_set, is_train, transform, bbox_map = None):
        rule = (lambda x: x < 100) if is_train else (lambda x: x >= 100)
        train_filtered_idx = [
            i for i, x in enumerate(data_set.targets) if rule(x)
        ]
        np.random.shuffle(train_filtered_idx)
        self.bbox = []
        self.data = np.array(data_set.samples)[train_filtered_idx].tolist()
        self.targets = np.array(data_set.targets)[train_filtered_idx].tolist()
        self.transform = transform
        self.class_to_idx = data_set.class_to_idx
        # map bounding boxes
        if bbox_map:
          for i in range(len(self.data)):
            self.bbox.append(bbox_map[self.data[i][0]])
        msg_str = 'train' if is_train else 'test'
        print(f'=> Number of images in {msg_str} : {len(self.data)}')
        print(f'=> Number of labels in {msg_str} : {len(self.targets)}')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, target = self.data[index][0], self.targets[index]
        image = Image.open(image_path)
        image = image.convert('RGB').crop(self.bbox[index]).resize((256,256))
        if self.transform is not None:
            image = self.transform(image)
        return image, target


# Define train and test transforms
train_transforms = transforms.Compose([ transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
~~~

### Train Dataset Object

Now, we create the `train_dataset` object. For the `DataLoader`, the `sampler` argument is set to `MPerClassSampler`, which is available in `pytorch_metric_learning`. At every iteration, this will return 'm' samples per class, assuming that the batch size is a multiple of 'm'.


~~~python
# Initialize train dataset and dataloader

# Train dataset
train_dataset = ClassDisjointDataset(cub_set, True, transform=train_transforms, bbox_map=image_bbox_map)

# Using MPerClassSampler to ensure each batch has a 'M' samples of each class, must --> batch_size % m = 0
sampler = MPerClassSampler(train_dataset.targets,config['m_per_sample'],batch_size=config['batch_size'])#,length_before_new_iter=50000)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler,num_workers=config['num_workers'])
~~~

## 5. Image Retrieval Model 

The Image Retrieval model `ImageEmbedder` is derived from the [Combination of Multiple Global Descriptors for Image Retrieval (CGD)](https://arxiv.org/abs/1903.10663v3) paper. CGD exploits multiple global descriptors to achieve an ensemble effect. Ensembling and combining multiple descriptors can help generate a richer embedding for the task of image retrieval.

In the original implementation, CGD uses `resnet50` as the backbone where stage 3 downsampling is removed. This modification results in a 14×14 sized feature map at the end for an input size of 224 × 224. However, for simplicity, we use `resnet50` without any modification, which produces a feature map of size 7x7.

### Network Architecture

<div align="center" markdown="1">
<img src="https://raw.githubusercontent.com/leftthomas/CGD/master/results/structure.png" alt="Example of Sparrow Images."/>
<figcaption align="center" markdown="1">
Figure 3. Image Retrieval Model Architecture. Image source: [CGD paper](https://arxiv.org/pdf/1903.10663v3.pdf)
</figcaption>
</div>

The main module has multiple branches that output each image representation by using different global descriptors on the last convolutional layer. The features from these branches are then combined (either by adding or concatenating) to generate a 1536-dimensional embedding vector.

The model also includes an auxiliary classification module that fine-tunes the CNN backbone using a classification loss.

~~~python

# set_bn_eval function is used from https://github.com/leftthomas/CGD
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()

class GEMPool(nn.Module):

    def __init__(self,p):
      super().__init__()
      self.p = p

    def forward(self,x):
      # ref: https://github.com/leftthomas/CGD
      sum_value = x.pow(self.p).mean(dim=[-1, -2])
      x_ = torch.sign(sum_value) * (torch.abs(sum_value).pow(1.0 / self.p))
      return x_


class ImageEmbedder(nn.Module):

    def __init__(self, d=1024,num_classes=100):
      super().__init__()

      # backbone
      self.backbone = models.resnet50(weights='IMAGENET1K_V2')
      num_features = self.backbone.fc.in_features

      # replace avgpool with Identity
      self.backbone.avgpool = nn.Identity()

      # set fc to identity
      self.backbone.fc = nn.Identity()

      # embedding layers,
      self.gd_1 = nn.Sequential(GEMPool(p=3),nn.Linear(num_features, d // 2, bias=False), nn.LayerNorm(d//2))
      self.gd_2 = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)),nn.Flatten(),nn.Linear(num_features, d // 2, bias=False), nn.LayerNorm(d//2))

      # classification layer
      self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(num_features, num_classes))
      self.is_mode_training = True

    def forward(self,x):
      x = self.backbone(x)
      x = torch.reshape(x,(x.shape[0],2048,7,7))
      classes = self.fc(x)
      embedding = F.normalize(torch.cat([self.gd_2(x),self.gd_1(x)], dim=-1), dim=-1)
      if self.is_mode_training:
        return embedding, classes
      return embedding
~~~

In the above `ImageEmbedder` class, we use two global descriptors. First, `self.gd_1` uses GeM (`GEMPool`) followed by dimensionality reduction using `nn.Linear`. Second, `self.gd_2` uses `nn.AdaptiveAvgPool2d` followed by `nn.Linear`. Finally, `self.fc` is used as a classification branch.

## 6. Training

Following the paper, the initial learning rate is set to 1e-4. For the auxiliary classification loss, [Temperature scaling](https://arxiv.org/abs/1706.04599) in softmax cross-entropy loss and [label smoothing](https://arxiv.org/abs/1512.00567) are used, as they have been proven to be helpful for the training process.

For the triplet loss, the `TripletMarginLoss` module from `pytorch_metric_learning` is utilized. The `TripletMarginLoss` computes all possible triplets within the batch, based on the labels passed into it. Anchor-positive pairs are formed by embeddings that share the same label, and anchor-negative pairs are formed by embeddings that have different labels.

Additionally, it is helpful to add a miner. In the following code, the miner finds positive and negative pairs that it deems particularly difficult. In the following snippet, `train_epoch` is a function for training a model for one epoch.

~~~python
def load_model(config_):
  # Model Init
  model_config = config_['model']
  model = ImageEmbedder(d=model_config['feature_dim'])

  return model
~~~

~~~python
# Training function
def train_epoch(net, optim, train_data_loader, classification_loss, margin_loss, mining_func, epoch, num_epochs, T):

    net.train()
    net.apply(set_bn_eval)
    len_dataloader = len(train_data_loader)
    idx = 0
    total_loss, total_correct, total_num = 0, 0, 0

    for inputs, labels in train_data_loader:

        # fetch batch
        inputs, labels = inputs.cuda(), labels.cuda()

        # forward pass
        features, classes = net(inputs)

        # classifcation loss with temperature scaling
        class_loss = classification_loss(classes/T, labels)

        # triplet loss, mining func will mine triplets in the batch
        indices_tuple = mining_func(features, labels)
        feature_loss = margin_loss(features, labels, indices_tuple)
        loss = class_loss + feature_loss

        # optimizer step
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Accuracy calcuation
        pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        if idx % 10 == 0:
          print('ITER:{:10} Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}% Number of mined triplets = {} Alpha = {}'
                                 .format(idx,epoch+1, num_epochs, total_loss / total_num, total_correct / total_num * 100, mining_func.num_triplets,alpha))
        idx = idx+1

    return total_loss / total_num, total_correct / total_num * 100
~~~

The training loops is follows,

~~~python
def train(config):

  ### Training related Suff ###
  training_config = config['training']

  # load model
  model = load_model(config)
  model.to(DEVICE)

  # Optimizer/ LR and LR Scheduler
  optimizer = Adam(model.parameters(), lr=1e-4)
  lr_scheduler = None
  if training_config['num_epochs'] > 1:
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.6 * training_config['num_epochs']), int(0.8 * training_config['num_epochs'])], gamma=0.1)
  class_criterion = nn.CrossEntropyLoss(label_smoothing=training_config['smoothing'])

  ### pytorch-metric-learning , TripletMarginLoss and TripletMiner ###
  reducer = reducers.ThresholdReducer(low=0)
  margin_loss_func = losses.TripletMarginLoss(margin=training_config['margin'], reducer=reducer)
  mining_func = miners.TripletMarginMiner(
      margin=training_config['margin'], type_of_triplets="hard"
  )

  ### Training Loop ###
  for epoch in range(training_config['num_epochs']):
    loss, acc = train_epoch(model,optimizer,train_loader,class_criterion,margin_loss_func,mining_func,epoch,training_config['num_epochs'],training_config['temprature'])
    if lr_scheduler is not None:
      lr_scheduler.step()
    torch.save(model.state_dict(), config['model_save_path'])

if config['mode'] == 'train':
  train(config)
~~~

## 7. Evaluation 

To evaluate the performance of the model, we perform quantitative and qualitative analysis of the model.

### Quantitative Analysis

For quantitative analysis, `precision_at_1` is calculated using `pytorch_metric_learning`. The `AccuracyCalculator` class in `pytorch_metric_learning`  computes several accuracy metrics given a query and reference embeddings. 

Create an object of `AccuracyCalculator`,

~~~python
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
~~~

then the test function is as follows,

~~~python
def get_all_embeddings(model, dataset):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def test( model, query_set, reference_set=None, accuracy_calculator=None, ref_includes_query=False, ranks=[1,2,4,8]):
  query_embeddings, query_labels = get_all_embeddings(model, query_set)
  query_labels = query_labels.squeeze(1)
  ref_embeddings = None
  ref_labels = None

  if reference_set:
    ref_embeddings, ref_labels = get_all_embeddings(model, reference_set)
    ref_labels = ref_labels.squeeze(1)

  if accuracy_calculator is None:
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

  accuracies = accuracy_calculator.get_accuracy(
        query_embeddings, query_labels, ref_embeddings, ref_labels, ref_includes_query
    )

  return accuracies
~~~
Load model for evaluation,

~~~python
### Load model for the evaluation ###
def get_eval_model(config_, device, checkpoint):
  eval_model = load_model(config_)
  eval_model.to(device)
  # load checkpoint
  eval_model.load_state_dict(torch.load(checkpoint,map_location=device))
  eval_model.is_mode_training=False
  eval_model.eval()
  return eval_model

eval_model = get_eval_model(config,
                            DEVICE,
                            config['model_save_path'])
~~~

Initialize `test_dataset`,

~~~python
# Load test dataset
test_dataset = CubDisjointDataset(cub_set, False, transform=test_transforms, bbox_map=image_bbox_map)
~~~

Finally, run the test after training the model for 2 epochs. We obtained a `precision_at_1` value of 0.72.

~~~python
# run test on wholse set, in this case ref_includes_query
ranks = [1,2,4,8]
accs = test(eval_model, test_dataset, accuracy_calculator=accuracy_calculator,ref_includes_query=True,ranks=ranks)

def print_metric(metrics_, ranks):
  print('\n')
  print ('---------------Performance Metric-----------------')
  print('\n')
  print (f"=> precision_at_1 : {metrics_['precision_at_1']}")
  print('\n')
  print('--------------------------------------------------')
  print('\n')

print_metric(accs,ranks)
~~~


### Qualitative Analysis

For qualitative analysis, we employ the `inference` module from `pytorch_metric_learning`. This module takes a query image as input, allowing the model to predict and retrieve the nearest images. Subsequently, we visualize these images alongside their respective ground truth class labels.

### InferenceModel wrapper from `pytorch_metric_learning`

~~~python
match_finder = MatchFinder(distance=LpDistance(), threshold=0.7)
inference_model = InferenceModel(eval_model, match_finder=match_finder)
inference_model.train_knn(test_dataset)
~~~

Generate a random query image index from the `test_dataset`, then fetch the k=6 nearest neighbors. Run the following cell multiple times to test random images from the test set. We discard the first prediction, as our reference contains the query image; the nearest prediction will be the query image.

~~~python
images_with_labels = []

k = 6
query_id = np.random.randint(len(test_dataset))
print('=> Query ID {}'.format(query_id))
# Fetch query image
img = test_dataset[query_id][0].unsqueeze(0)
label = test_dataset[query_id][1]

print(f"=> Query image class id: {label}")
print(f"=> Query image class name : {class_id_to_name[label]}")

# For plot
show_query_image = inv_normalize(test_dataset[query_id][0])
show_query_image = show_query_image.numpy().transpose((1, 2, 0))
images_with_labels.append((show_query_image,f'Query - {class_id_to_name[label]}'))

# featch k nearest images
distances, indices = inference_model.get_nearest_neighbors(img, k=k)
indices = indices.cpu()[0][1:]
distances = distances.cpu()[0][1:]
nearest_imgs = [test_dataset[i][0] for i in indices]

# show nearest images
nearest_labels = [class_id_to_name[test_dataset[i][1]] for i in indices]
print(' | '.join(nearest_labels))
print("nearest images")


for idx in indices:
  img = test_dataset[idx][0]
  label = class_id_to_name[test_dataset[idx][1]]
  img = inv_normalize(img)
  img = img.numpy().transpose((1, 2, 0))
  images_with_labels.append((img,label))

plot_image_grid(images_with_labels, grid_columns=6, figsize=(20, 5), fontsize=10)
~~~

### Visualization

Finally, let us visualize a few outputs from the model. The first image in each row is the query image, and the rest of the images are retrieved images by the model.


<div align="center" markdown="1">
<img src="/assets/images/post-0008/Query_image_predictions_1.png" alt="Visualization of model outputs"/>
</div>
<div align="center" markdown="1">
<img src="/assets/images/post-0008/Query_image_predictions_2.png" alt="Visualization of model outputs"/>
</div>
<div align="center" markdown="1">
<img src="/assets/images/post-0008/Query_image_predictions_3.png" alt="Visualization of model outputs"/>
</div>
<div align="center" markdown="1">
<img src="/assets/images/post-0008/Query_image_predictions_4.png" alt="Visualization of model outputs"/>
</div>
<div align="center" markdown="1">
<img src="/assets/images/post-0008/Query_image_predictions_5.png" alt="Visualization of model outputs"/>
</div>
<div align="center" markdown="1">
<img src="/assets/images/post-0008/Query_image_predictions_sparrow.png" alt="Visualization of model outputs"/>
</div>
<div align="center" markdown="1">
<figcaption align="center" markdown="1">
Figure 4. Qualitative Analysis of the Image Retrieval Model. 
</figcaption>
</div>

The complete source code and Colab notebook for this tutorial are available on my [GitHub repo](https://github.com/ranipakeyur/image_retrieval/blob/main/pytorch_tutorial_content_based_image_retrieval_v1.ipynb).

## 8. References

1. [Combination of Multiple Global Descriptors for Image Retrieval](https://arxiv.org/pdf/1903.10663v3.pdf)
2. [https://github.com/leftthomas/CGD](https://github.com/leftthomas/CGD)
3. [https://github.com/KevinMusgrave/pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

