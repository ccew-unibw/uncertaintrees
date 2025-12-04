"""
Functions to generate semi-automated cluster assignments of grid cells for the local models.
Implemented are DBSCAN and HDBSCAN approaches, as well as cluster assignment based on UN
statistical regions and support for the pipeline's test mode.
"""

import os
from datetime import date

import numpy as np
import pandas as pd
from pyproj import CRS
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
from shapely import MultiPoint, Point, box
from shapely.constructive import concave_hull
from shapely.ops import nearest_points

from src.utils.conversion import latlon_to_pgid, pgid_to_latlon, get_month_id
from src.utils.data_prep import create_dummy_data

# TODO: the dbscan and hdbscan functions share quite a bit of code - could be refactored


def make_clusters_unregions(
    df: pd.DataFrame, s: float = 0.5, visualize_clusters: bool = False
) -> pd.DataFrame:
    """
    Information on the source files used for country/region matching (file paths are hardcoded):
    Iso3 Matching: GeoBoundaries (https://www.geoboundaries.org/) Comprehensive Global
        Administrative Zones (CGAZ) ADM0 - not in repo, needs to be downloaded to regenerate.
    Areas: United Nations Statistics Division: Standard country or area codes for statistical
        use (M49) - Sub-regions downloaded from:
        https://unstats.un.org/unsd/methodology/m49/overview/

    Args:
        df (pd.DataFrame): Dataframe with data used for clustering
        s (float, optional): grid size (should not be changed unless for other applications)
        visualize_clusters (bool, optional): whether to also plot a graphic with the clusters

    Returns:
        pandas DataFrame with cluster assignments
    """

    def get_neighbors_indices(id):
        y, x = pgid_to_latlon(id)
        ys = [y - s, y, y + s]
        xs = [x - s, x, x + s]
        index_list = [
            (ys[0], xs[0]),
            (ys[0], xs[1]),
            (ys[0], xs[2]),
            (ys[1], xs[0]),
            (ys[1], xs[2]),
            (ys[2], xs[0]),
            (ys[2], xs[1]),
            (ys[2], xs[2]),
        ]
        # handle edge cases
        index_list = [id for id in [latlon_to_pgid(i[0], i[1]) for i in index_list] if id in pgids]
        return index_list

    def assign_country(pgid):
        lat, lon = pgid_to_latlon(pgid)
        geometry = box(lon - s / 2, lat - s / 2, lon + s / 2, lat + s / 2)
        try:
            clipped = countries.clip(geometry)
        except Exception as e:  # should not happen anymore
            print(pgid, e)
            return "exception"
        if clipped.empty:  # this means if not touching any geometry
            return "not_assigned"
        elif len(clipped) == 1:
            return clipped.iloc[0].shapeGroup
        else:
            # reprojecting to albers equal area equidistant projection for area calculations
            lat_1 = lat - s / 3
            lat_2 = lat + s / 3
            aea_proj = CRS(
                f'+proj=aea +ellps="WGS84" +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat} +lon_0={lon} +units=m'
            )
            # get area
            clipped["area_size"] = clipped.to_crs(aea_proj).area
            # get index of the largest area in cell and assign its name to grid
            i = clipped.area_size.idxmax()
            return countries.loc[i]["shapeGroup"]

    def region_check(region, pgid):
        """
        in case of unclear region assignments, use the region where the majority of neighbors is
        """
        if pd.isna(region):
            assert np.isnan(region)
            neighbors_pgids = get_neighbors_indices(pgid)
            df_buffer = df_clusters.loc[df_clusters.priogrid_gid.isin(neighbors_pgids)]
            unique_regions = df_buffer.region.unique()
            if len(unique_regions) == 1:
                return unique_regions[0]
            else:
                counts = df_buffer.groupby("region")["priogrid_gid"].count()
                if np.unique(counts, return_counts=True)[1][-1] > 1:
                    raise NotImplementedError(pgid)
                else:
                    return counts.idxmax()
        else:
            return region

    if os.path.exists("data/region_matching.parquet"):
        df_clusters = pd.read_parquet("data/region_matching.parquet")
    else:
        pgids = df.index.get_level_values("priogrid_gid").unique()
        # geoboundaries for country matching
        countries = gpd.read_file("data/geoBoundariesCGAZ_ADM0.gpkg", crs="4326")
        countries.geometry = (
            countries.geometry.make_valid()
        )  # takes a bit but avoids annoying erros
        # UNSD regions - using Intermedia Regions
        regions = pd.read_csv(
            "data/UNSDMethodology.csv",
            delimiter=";",
            usecols=[
                "Country or Area",
                "ISO-alpha3 Code",
                "Intermediate Region Name",
                "Sub-region Name",
            ],
        )
        regions["region"] = regions.apply(
            lambda x: x["Sub-region Name"]
            if pd.isna(x["Intermediate Region Name"])
            else x["Intermediate Region Name"],
            axis=1,
        )
        regions = regions.rename(
            columns={"Country or Area": "country", "ISO-alpha3 Code": "iso3"}
        ).drop(columns=["Intermediate Region Name", "Sub-region Name"])

        # match grid with countries - majority rule
        df_clusters = pd.DataFrame(data=pgids)
        # this leaves some grid cells unassigned but we fix it when assigning areas
        df_clusters["iso3"] = df_clusters.priogrid_gid.progress_apply(assign_country)
        df_clusters = df_clusters.merge(regions, how="left", left_on="iso3", right_on="iso3")
        df_clusters.region = df_clusters.apply(
            lambda x: region_check(x["region"], x["priogrid_gid"]), axis=1
        )

        # some manual corrections are required at the outside borders around the Middle East
        # -> everything is grouped to Western Asia (also Iran, which would be alone otherwise)
        df_clusters = df_clusters.replace(
            to_replace=[
                "Southern Europe",
                "Eastern Europe",
                "Central Asia",
                "Southern Asia",
            ],
            value="Western Asia",
        )
        df_clusters["region"] = df_clusters["region"].str.replace(" ", "_").lower() # type: ignore

        # save
        df_clusters = (
            df_clusters.drop(columns=["iso3", "country"])
            .set_index("priogrid_gid")
            .rename(columns={"region": "cluster"})
        )
        df_clusters.to_parquet("region_matching.parquet")

    if visualize_clusters:
        countries = gpd.read_file("data/geoBoundariesCGAZ_ADM0.gpkg", crs="4326")
        df_test = df_clusters.copy()
        df_test["coords"] = list(df_test.reset_index().priogrid_gid.apply(pgid_to_latlon))
        df_test["geometry"] = df_test.coords.apply(
            lambda x: box(x[1] - 0.25, x[0] - 0.25, x[1] + 0.25, x[0] + 0.25)
        )
        gdf = gpd.GeoDataFrame(df_test, geometry=df_test.geometry)
        views_bounds = gdf.total_bounds

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        gdf.plot(column="cluster", edgecolor="None", ax=ax, cmap="turbo")
        countries.plot(facecolor="None", edgecolor="black", ax=ax)
        ax.set_xlim(views_bounds[0], views_bounds[2])
        ax.set_ylim(views_bounds[1], views_bounds[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return df_clusters


def make_clusters_dbscan(
    df: pd.DataFrame, cluster_kwargs: dict, visualize_clusters: bool = False
) -> pd.DataFrame:
    """
    Semi-automated clustering with DBSCAN based on spatial distribution of grid cells with
    violence across input data (disregarding time) and manually set parameters for the algorithm.
    Grid cells without violence or not initially assigned are assigned based on the distance to
    polygons drawn around identified clusters.

    Args:
        df (pd.DataFrame): Dataframe with data used for clustering
        cluster_kwargs (dict): parameters for clustering algorithm
        visualize_clusters (bool): whether to also plot a graphic with the clusters

    Returns:
        pandas DataFrame with cluster assignments
    """
    ## Identify pgids with fatalities and create clust_df
    clust_df = (
        df.loc[: get_month_id(date(2017, 10, 31)), :]
        .groupby("priogrid_gid")["ged_sb"]
        .sum()
        .to_frame()
    )
    lat, lon = np.vectorize(pgid_to_latlon)(clust_df.index.to_list())
    clust_df = clust_df.assign(lat=lat, lon=lon)

    # setup lookup dict
    cluster_assignments = dict.fromkeys(clust_df.index)

    points_violence = clust_df[clust_df["ged_sb"] > 0][["lon", "lat"]].to_numpy()
    pgids_violence = clust_df[clust_df["ged_sb"] > 0].index.to_numpy()

    def create_clusters_dbscan():
        # initial cluster assignments
        cluster = DBSCAN(**cluster_kwargs).fit(points_violence)

        labels = cluster.labels_
        # Store all results in the result dict that are not -1
        for idx, label in enumerate(labels):
            cluster_assignments[pgids_violence[idx]] = label

        # Create polygons around cluster grid cells
        polys = []
        for label in set(labels):
            # Skip noise clusters
            if label == -1:
                continue
            mask = labels == label  # Get all points belonging to the existing cluster
            polys.append(concave_hull(MultiPoint(points_violence[mask]), ratio=0.25))
        return labels, polys

    def assign_zeros_to_clusters():
        # Now, create a distance matrix measuring the shortest distance of every point outside of the polygons
        # to the polygon borders.

        # Select all points that are not inside any cluster
        # That is basically all points with the target ariable equal to 0 and all points with a label of -1
        coords = np.concatenate(
            (
                clust_df[clust_df["ged_sb"] == 0][["lon", "lat"]].to_numpy(),
                points_violence[labels_violence == -1],
            )
        )
        pgids = np.concatenate(
            (
                clust_df[clust_df["ged_sb"] == 0].index.to_numpy(),
                pgids_violence[labels_violence == -1],
            )
        )
        for pgid, coord in zip(pgids, coords):
            point = Point(coord)
            min_distance = float("inf")  # Initialize with a very large value
            for poly_idx, poly in enumerate(polys):
                nearest = nearest_points(poly.boundary, point)[
                    0
                ]  # Get the nearest point on the polygon's boundary
                distance = point.distance(
                    nearest
                )  # Calculate the distance between the point and the nearest point
                if distance < min_distance:
                    min_distance = distance
                    cluster_assignments[pgid] = poly_idx
        return

    # both of those save their results in cluster_assignments
    labels_violence, polys = create_clusters_dbscan()
    assign_zeros_to_clusters()

    clust_df["cluster"] = pd.Series(cluster_assignments)

    if visualize_clusters:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        da = clust_df.set_index(["lat", "lon"])["cluster"].to_xarray()
        da.plot(
            ax=ax,
            cmap="nipy_spectral",
            vmin=clust_df["cluster"].min(),
            vmax=clust_df["cluster"].max(),
        )  # type: ignore
        plt.tight_layout()
        plt.show()
    return clust_df


def make_clusters_hdbscan(
    df: pd.DataFrame,
    min_months: int,
    cluster_kwargs: dict,
    visualize_clusters: bool = False,
) -> pd.DataFrame:
    """
    Semi-automated clustering with HDBSCAN based on spatial distribution of grid cells with
    violence across input data (disregarding time) and manually set parameters for the algorithm.
    Smaller clusters are merged with their nearest neighbors based on centroid distance if they
    do not include at least [min_months] grid months with violence. Grid cells without violence
    or not initially assigned are assigned based on the distance to polygons drawn around
    identified clusters.

    Args:
        df (pd.DataFrame): Dataframe with data used for clustering
        min_months (int): Minimum number of grid cell within a cluster for it not to be merged
        cluster_kwargs (dict): parameters for clustering algorithm
        visualize_clusters (bool): whether to also plot a graphic with the clusters

    Returns:
        pandas DataFrame with cluster assignments
    """

    def cluster_polys(ratio=0.5):
        polys = []
        ls = []
        for label in set(labels):
            # Skip noise clusters
            if label == -1:
                continue
            else:
                mask = labels == label  # Get all points belonging to the existing cluster
                # poly = MultiPoint(points[mask]).convex_hull
                polys.append(concave_hull(MultiPoint(points_violence[mask]), ratio=ratio))
                ls.append(label)
        gdf_polys = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(polys), data=pd.Series(ls, name="index")
        ).set_index("index")
        return gdf_polys

    def assign_zeros_to_clusters():
        # Now, create a distance matrix measuring the shortest distance of every point outside of the polygons
        # to the polygon borders.

        # Select all points that are not inside any cluster
        # That is basically all points with the target variable equal to 0 and all points with a label of -1
        coords = np.concatenate(
            (
                clust_df[clust_df["ged_sb"] == 0][["lon", "lat"]].to_numpy(),
                points_violence[labels == -1],
            )
        )
        pgids = np.concatenate(
            (
                clust_df[clust_df["ged_sb"] == 0].index.to_numpy(),
                pgids_violence[labels == -1],
            )
        )
        for pgid, coord in zip(pgids, coords):
            point = Point(coord)
            min_distance = float("inf")  # Initialize with a very large value
            for cluster, poly in gdf_polys.geometry.items():
                nearest = nearest_points(poly.boundary, point)[
                    0
                ]  # Get the nearest point on the polygon's boundary
                distance = point.distance(
                    nearest
                )  # Calculate the distance between the point and the nearest point
                if distance < min_distance:
                    min_distance = distance
                    clust_df.at[pgid, "cluster"] = cluster # type: ignore
        return

    ## Identify pgids with fatalities and create clust_df
    clust_df = (
        df.loc[: get_month_id(date(2017, 10, 31)), :]
        .groupby("priogrid_gid")["ged_sb"]
        .sum()
        .to_frame()
    )
    lat, lon = np.vectorize(pgid_to_latlon)(clust_df.index.to_list())
    clust_df = clust_df.assign(lat=lat, lon=lon)

    points_violence = clust_df[clust_df["ged_sb"] > 0][["lon", "lat"]].to_numpy()
    temp = (
        (df.loc[: get_month_id(date(2017, 10, 31)), :]["ged_sb"] != 0).groupby("priogrid_gid").sum()
    )
    weights = temp.loc[clust_df[clust_df["ged_sb"] > 0].index].values
    pgids_violence = clust_df[clust_df["ged_sb"] > 0].index.to_numpy()

    # manually tuned clusters for coverage of most visible cluster so there is something to assign to
    cluster = HDBSCAN(**cluster_kwargs).fit(points_violence)
    labels = cluster.labels_

    gdf_polys = cluster_polys()

    # calculate closest polygons for each based on centroids (this means smaller clusters are closer to each other)
    dist = []
    for poly in gdf_polys.geometry:
        distances = np.array(gdf_polys.centroid.distance(poly.centroid).sort_values().index[1:])
        dist.append(distances)
    gdf_polys["distances"] = dist

    # merge clusters which cover less than the specified minimum number of priogrid months to neighboring
    problem_groups = [c for c in np.unique(labels)[1:] if weights[labels == c].sum() < min_months]
    for c in problem_groups:
        mask = labels == c
        if weights[mask].sum() > min_months:
            continue
        labels[mask] = gdf_polys.loc[c].distances[0]

    # clean up clusters
    replace_map = {val: idx - 1 for idx, val in enumerate(np.unique(labels))}
    labels = np.vectorize(lambda x: replace_map[x])(labels)

    # setup lookup dict and add cluster info to clust_df
    cluster_assignments = dict.fromkeys(clust_df.index)
    for idx, label in enumerate(labels):
        cluster_assignments[pgids_violence[idx]] = label
    clust_df["cluster"] = pd.Series(cluster_assignments)

    # re-calculate polygons based on merged clusters before assigning non-assigned grids
    gdf_polys = cluster_polys()
    assign_zeros_to_clusters()

    clust_df["cluster"] = clust_df["cluster"].astype(int)

    if visualize_clusters:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        da = clust_df.set_index(["lat", "lon"])["cluster"].to_xarray()
        da.plot(
            ax=ax,
            cmap="nipy_spectral",
            vmin=clust_df["cluster"].min(),
            vmax=clust_df["cluster"].max(),
        )  # type: ignore
        gdf_polys.plot(ax=ax, facecolor="None", edgecolor="gray")
        plt.tight_layout()
        plt.show()
    return clust_df


def make_clusters_test_pgm() -> pd.DataFrame:
    """clusters for pipeline test mode"""
    df = create_dummy_data(2017)
    clust_df = pd.DataFrame(index=df.index.levels[1])  # type: ignore
    clust_df["cluster"] = clust_df.reset_index().priogrid_gid % 2
    return clust_df
