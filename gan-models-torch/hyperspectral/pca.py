import matplotlib.pyplot as plt
import numpy as np,os,sys
from scipy.io import loadmat
import rasterio as rio
from processor import Processor
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_pixels(scores_pca, num_components):
    wcss = []

    for i in range(1,num_components+1):
        kmeans_pca = KMeans(n_clusters= i, init = 'k-means++', random_state = 42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)

    plt.figure(figsize=(10,8))
    plt.plot(range(1,num_components+1))
    plt.xlabel('number of clusters')
    plt.ylabel('WCSS')
    plt.title('K-Means with PCA Clustering')
    plt.show()


def extract_pixels(X):
    q = X.reshape(-1, X.shape[2])
    df = pd.DataFrame(data = q)
    #df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis=1)
    df.columns= [f'band{i}' for i in range(1, 1+X.shape[2])]
    df.to_csv('Dataset.csv')
    return df

def convert_PCA(hsi, num_components, flattened_output=True):
    
    df = extract_pixels(hsi)
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)


    pca = PCA(n_components = num_components)
    pca.fit(df_std)
    scores_pca = pca.transform(df_std)
    print(scores_pca.shape)
    if flattened_output:
        return scores_pca
    else:
        return scores_pca.reshape(256,256,3)


def K_Means_Clustering(scores_pca, num_clusters):

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scores_pca)

    cluster_labels_image = cluster_labels.reshape(256, 256)

    plt.figure(figsize=(8, 6))
    plt.imshow(cluster_labels_image, cmap='viridis')  # Adjust the colormap as needed
    plt.colorbar(label='Cluster Label')
    plt.title('K-means Clustering Results')
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    plt.show()


if __name__ == "__main__":
    
    p = Processor()

    p.prepare_data(r'datasets/export_2/trainA/session_000_001k_048_snapshot_ref.tiff')

    print(p.hsi_data.shape)
    X= p.hsi_data
    
    X = p.hsi_data.reshape(-1, 51)
    
    
    covariance_matrix=np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    ind=np.arange(0,len(eigen_values),1)
    ind=[x for _,x in sorted(zip(eigen_values,ind))]
    ind=ind[::-1]
    eigen_values1=eigen_values[ind]
    eigen_vectors1=eigen_vectors[:,ind]
    eigen_vectors1=eigen_vectors1[:,:3]

        
    # y=(eigen_vectors1.T).dot(X.T) 
    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")
    # ax.scatter3D(y[0,:], y[1,:], y[2,:], color = "green")
    # plt.title("3D Projection of Eigen Vectors")


        
    projection_matrix = (eigen_vectors.T[:][:3]).T
    print(projection_matrix)

    # eigen_values1=eigen_values1[:10]
    # x=np.arange(0,len(eigen_values1),1)
    # plt.plot(x,eigen_values1,marker = 'o')
    #plt.show()

    X = X.reshape(256,256,51)

    # fig = plt.figure(figsize = (12, 6))

    # for i in range(1, 1+6):
    #     fig.add_subplot(2,3, i)
    #     q = np.random.randint(X.shape[2])
    #    # plt.imshow(X[:,:,q], cmap='nipy_spectral')
    #     plt.axis('off')
    #     plt.title(f'Band - {q}')



    # print(ev)

    # plt.figure(figsize=(12, 6))
    # plt.plot(np.cumsum(ev))
    # plt.xlabel('Number of components')
    # plt.ylabel('Cumulative explained variance')


    # plt.show()


    # pca = PCA(n_components = 3)
    # pca.fit(df_std)
    # scores_pca = pca.transform(df_std)
    # print(scores_pca.shape)

    # num_clusters = 3  # Specify the number of clusters
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # cluster_labels = kmeans.fit_predict(scores_pca)

    # cluster_labels_image = cluster_labels.reshape(256, 256)



    # plt.figure(figsize=(8, 6))
    # plt.imshow(cluster_labels_image, cmap='viridis')  # Adjust the colormap as needed
    # plt.colorbar(label='Cluster Label')
    # plt.title('K-means Clustering Results')
    # plt.xlabel('Pixel Column')
    # plt.ylabel('Pixel Row')
    # plt.show()
# #     dt = pca.fit_transform(df.iloc[:, :-1].values)
# #     q = pd.concat([pd.DataFrame(data = dt), pd.DataFrame(data = y.ravel())], axis = 1)
# #     q.columns = [f'PC-{i}' for i in range(1,4)]+['class']

# #     print(q)

# #     fig = plt.figure(figsize = (20, 10))

# #     for i in range(1, 1+3):
# #         fig.add_subplot(2,4, i)
# #         plt.imshow(q.loc[:, f'PC-{i}'].values.reshape(256, 256), cmap='nipy_spectral')
# #         plt.axis('off')
# #         plt.title(f'Band - {i}')
# #    # plt.show()

    #cluster_pixels(scores_pca, 60)
    scores_PCA = convert_PCA(p.hsi_data, 3)
    K_Means_Clustering(scores_PCA, 3)
