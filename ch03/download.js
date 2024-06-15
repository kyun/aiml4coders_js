const url =
  "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip";
const path = "./horse-or-human.zip";
const fs = require("fs");
const axios = require("axios");
const { pipeline } = require("stream");
const unzipper = require("unzipper");

const download = async (url, path) => {
  const response = await axios({
    url,
    method: "GET",
    responseType: "stream",
  });

  pipeline(response.data, fs.create);
};

download(url, path);
// The download function downloads the file from the given URL and saves it to the given path.

const extract = async (path) => {
  fs.create.ReadStream.pipe(unzipper.Extract({ path: "./data" }));
};

extract(path);
