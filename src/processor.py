import pandas as pd


class Processor:
    def __init__(self, data) -> None:
        """
        Initialize Processor object with given data.

        Args:
            data (pandas.DataFrame): input data to be processed
        """
        self.data = data

    def __extract_titles(self, s: str) -> str:
        """
        Extract titles from a given string.

        Args:
            s (str): input string

        Returns:
            str: extracted title
        """
        return s.extract(" ([A-Za-z]+)\.", expand=False)  # noqa: W605

    def __replace_rare_titles(self, s: pd.Series) -> pd.Series:
        """
        Replace rare titles in a given pandas Series.

        Args:
            s (pandas.Series): input Series

        Returns:
            pandas.Series: Series with rare titles replaced by "Rare"
        """
        return s.replace(
            [
                "Lady",
                "Countess",
                "Capt",
                "Col",
                "Don",
                "Dr",
                "Major",
                "Rev",
                "Sir",
                "Jonkheer",
                "Dona",
            ],
            "Rare",
        )

    def __replace_miss_variants(self, s: pd.Series) -> pd.Series:
        """
        Replace "Mlle" and "Ms" with "Miss" in a given pandas Series.

        Args:
            s (pandas.Series): input Series

        Returns:
            pandas.Series: Series with "Mlle" and "Ms" replaced by "Miss"
        """
        return s.replace(["Mlle", "Ms"], "Miss")

    def __replace_mrs_variants(self, s: pd.Series) -> pd.Series:
        """
        Replace "Mme" with "Mrs" in a given pandas Series.

        Args:
            s (pandas.Series): input Series

        Returns:
            pandas.Series: Series with "Mme" replaced by "Mrs"
        """
        return s.replace("Mme", "Mrs")

    def __title_mapping(self, s: pd.Series) -> int:
        """
        Map titles in a given pandas Series to numbers.

        Args:
            s (pandas.Series): input Series

        Returns:
            int: mapped number
        """
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        return s.map(title_mapping)

    def _compute_family_size(self) -> None:
        """
        Compute a new feature called "FamilySize" by adding "SibSp" and "Parch"
        """
        # new feature called "FamilySize" by adding "SibSp" and "Parch"
        self.data["FamilySize"] = self.data["SibSp"] + self.data["Parch"] + 1

    def _compute_is_alone(self) -> None:
        """
        Compute a new feature called "IsAlone" by checking if "FamilySize" is 0
        """
        # new feature called "IsAlone" by checking if "FamilySize" is 0
        self.data["IsAlone"] = 0
        self.data.loc[self.data["FamilySize"] == 0, "IsAlone"] = 1

    def _compute_title(self) -> None:
        """
        Compute a new feature called "Title" by extracting titles from "Name".
        Replace rare titles with "Rare".
        Replace "Mlle" and "Ms" with "Miss".
        Replace "Mme" with "Mrs".
        Map titles to numbers.
        Fill missing values in "Title" with 0.
        """
        # new feature called "Title" by extracting titles from "Name"
        self.data["Title"] = self.__extract_titles(self.data["Name"].str)
        # replace rare titles with "Rare"
        self.data["Title"] = self.__replace_rare_titles(self.data["Title"])
        # replace "Mlle" and "Ms" with "Miss"
        self.data["Title"] = self.__replace_miss_variants(self.data["Title"])
        # replace "Mme" with "Mrs"
        self.data["Title"] = self.__replace_mrs_variants(self.data["Title"])
        # map titles to numbers
        self.data["Title"] = self.__title_mapping(self.data["Title"])

    def _compute_age_band(self) -> None:
        # create a new feature called "AgeBand" by binning "Age"
        bins = [0, 2, 4, 13, 20, 110]
        labels = ["Infant", "Toddler", "Kid", "Teen", "Adult"]
        self.data["AgeBand"] = pd.cut(
            self.data["Age"], bins=bins, labels=labels, right=False
        )
        # map "AgeBand" to numbers
        age_mapping = {
            "Infant": 1,
            "Toddler": 2,
            "Kid": 3,
            "Teen": 4,
            "Adult": 5,
        }  # noqa: E501
        self.data["AgeBand"] = self.data["AgeBand"].map(age_mapping)
        self.data["AgeBand"] = self.data["AgeBand"].astype(float)

    def _compute_fare_band(self) -> None:
        # create a new feature called "FareBand" by binning "Fare"
        labels = [0, 1, 2, 3]
        self.data["FareBand"] = pd.qcut(
            self.data["Fare"], 4, labels=labels, duplicates="drop"
        )
        self.data["FareBand"] = self.data["FareBand"].astype(float)

    def _compute_sex(self) -> None:
        sex_mapping = {"male": 1, "female": 0}
        self.data["Sex"] = self.data["Sex"].map(sex_mapping)

    def process(self) -> pd.DataFrame:  # pragma: no cover # noqa: E501
        # compute "Sex"
        self._compute_sex()
        # compute "FamilySize"
        self._compute_family_size()
        # compute "IsAlone"
        self._compute_is_alone()
        # compute "Title"
        self._compute_title()
        # compute "AgeBand"
        self._compute_age_band()
        # compute "FareBand"
        self._compute_fare_band()
        # drop unused features
        self.data = self.data.drop(
            [
                "PassengerId",
                "Survived",
                "Name",
                "Age",
                "SibSp",
                "Parch",
                "Ticket",
                "Fare",
                "Cabin",
                "Embarked",
            ],
            axis=1,
        )

        return self.data
