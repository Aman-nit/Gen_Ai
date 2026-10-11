
const status = document.getElementById("status");

chrome.tabs.query(
  { active: true, lastFocusedWindow: true },
  (tabs) => {
    const tab = tabs[0];

    if (!tab || !tab.url) {
      status.textContent = "Cannot detect the current tab.";
      return;
    }

    try {
      const url = new URL(tab.url);
      let videoId = null;

      if (url.hostname === "youtu.be") {
        videoId = url.pathname.slice(1);
      } else if (
        ["youtube.com", "www.youtube.com", "m.youtube.com"]
          .includes(url.hostname)
      ) {
        videoId = url.searchParams.get("v");
      }

      if (videoId) {
        status.textContent =
          "Video detected! ID: " + videoId;
        console.log("Video URL:", tab.url);
        console.log("Video ID:", videoId);
      } else {
        status.textContent =
          "No video detected. Open a YouTube video page.";
      }
    } catch (error) {
      status.textContent = "Error: " + error.message;
    }
  }
);
