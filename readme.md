## [Mugiwara Pirates Image Classification](https://op-classifier.herokuapp.com)

Monkey D. Luffy is the series' main protagonist of One Piece, a young pirate who wishes to succeed Gol D. Roger, the deceased King of the Pirates, by finding his treasure, the "One Piece". Throughout the series, Luffy gathers himself a diverse crew, named the **Straw Hat Pirates**, including: the three-sword-wielding combatant Roronoa Zoro; the thief and navigator Nami; the cowardly marksman and inventor Usopp; the cook and martial artist Sanji; the anthropomorphic reindeer and doctor Tony Tony Chopper; the archaeologist Nico Robin; the cyborg shipwright Franky; the living skeleton musician Brook; and the fish-man helmsman Jinbe. 

The purpose of this project is for the user to upload an image of a member of Mugiwara Pirates and the model to provide a probability for the characters

## How to use

You just need to upload an image of a member of the Straw Hats
<details>
  <summary markdown="span">Example</summary>

<p align="center"><img align="center" src="img/ex.png" height="570px" width="620"/></p>

</details>


### Local requirements
First you need to unzip [`data.zip`](https://github.com/AlissonRP/OP-image-classification/blob/master/data.zip) and use the
```python
conda install --file requirements.txt
```
I really recommend you to use PyTorch + gpu, but for deploy I used it with cpu because the package size is smaller.
