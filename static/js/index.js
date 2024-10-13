window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/videos";
var NUM_INTERP_FRAMES = 200;

var json_data = {"5": {"vid": "a9nPldzIq-k_20", "label": "whistling"}, "10": {"vid": "K9tJRsDH03Y_18", "label": "whistling"}, "15": {"vid": "Q3PtqzPrUYA_390", "label": "clapping"}, "20": {"vid": "anlS-uikLj4_222", "label": "sneezing"},"25": {"vid": "pHT6uU1-7FY", "label": "sneezing"} ,"30": {"vid": "5aGq5zUc1S4_30", "label": "whistling"}, "35": {"vid": "1oR8FSr-eyw_000184_000194", "label": "singing"}, "40": {"vid": "bO1nArmOvmw_1", "label": "whistling"}, "45": {"vid": "AQqdts81Cfo_238", "label": "sneezing"}, "50": {"vid": "M_3Mk2nMydQ_30", "label": "cleaning floor"}, "55": {"vid": "mwa6EwHxvww_000007_000017", "label": "gargling"}, "60": {"vid": "qRWJ4Xs1ykE_30", "label": "sneezing"}, "65": {"vid": "7Wf6IA_oW-M_000147_000157", "label": "singing"}, "70": {"vid": "IkWW6YU4NMw_12", "label": "singing"}, "75": {"vid": "H1Hd6enwo6s_190", "label": "playing piano"}, "80": {"vid": "AtToCNcKuNM_200", "label": "playing flute"}, "85": {"vid": "Lq_NbIUNZ2s_14", "label": "shaving"}, "90": {"vid": "gyEX3QusCtU_278", "label": "shaving"}, "95": {"vid": "2Cp9ipnnvVs_000006_000016", "label": "cheering"}, "100": {"vid": "0hlTrRCscYg_000032_000042", "label": "motorcycling"}, "105": {"vid": "sDvyKbhX6oU_0", "label": "shaving"}, "110": {"vid": "UQaM2HQL8Hc_1", "label": "chopping wood"}, "115": {"vid": "5jdfAXApUkY_35", "label": "marching"}, "120": {"vid": "SjLXGtVMkR0_000224_000234", "label": "motorcycling"}, "125": {"vid": "zZhTQhXnWbc_1", "label": "playing cello"}, "130": {"vid": "vMw-P1CffuA_1", "label": "tractor digging"}, "135": {"vid": "Oy3w_wbRu5U_000382_000392", "label": "marching"}, "140": {"vid": "lYaUMsXoTkc_000074_000084", "label": "writing on blackboard"}, "145": {"vid": "HR3Hmo_Uq3Y_000117_000127", "label": "writing on blackboard"}, "150": {"vid": "I4_zfEBZRSg_588", "label": "swimming"}, "155": {"vid": "qsh7bh8AlR0_2", "label": "playing guitar"}, "160": {"vid": "rVu3BG3xzwc", "label": "cheering"}, "165": {"vid": "Y9fV2RowiYE_10", "label": "swimming"}, "170": {"vid": "kyOyrM4YktY_3", "label": "swimming"}, "175": {"vid": "yWfzoGN7c0I_0", "label": "playing cello"}, "180": {"vid": "JI9nRnJiP64_2", "label": "swimming"}, "185": {"vid": "Vm9tA8q1gOk_6", "label": "swimming"}, "190": {"vid": "JJBtLS-SrWk_4", "label": "playing guitar"}, "195": {"vid": "uO2bEU6Gaok_000165_000175", "label": "writing on blackboard"}}
var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i+= 10) {
    var path = INTERP_BASE + '/' + String(i) + '.mp4';
    // console.log("create", path)
    const video = document.createElement('video');
    video.controls = true;
    video.muted = false;
    video.height = 360; // in px
    video.width = 480; // in px
    video.src = path
    interp_images[i] = video;
    // console.log(json_data[String(i)])
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  
  // image.ondragstart = function() { return false; };
  // image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
  var paragraph = document.getElementById("video_category");
  paragraph.textContent="The label is:" + json_data[String(i)]['label'];

}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})
