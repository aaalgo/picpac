<template>
  <!-- Main content -->
  <section class='content'>
      <div class='row'>
        <div class="col-xs-12">
		<div class="box box-primary">
			<div class="box-header">
				  <h3 class="box-title">Stream Simulation Configuration</h3>
			</div>
		<div class="box-body">
        <div class="row">
        <div class="col-xs-3">
			  <label for="colorspace">Color Space:</label>
			  <select class="form-control" id="colorspace">
				<option>BGR</option>
				<option>HSV</option>
				<option>Lab</option>
				<option>Grayscale</option>
			  </select>
        </div>
        <div class="col-xs-3">
			  <label for="annotation">Annotation:</label>
			  <select class="form-control" id="annotation">
				<option>None</option>
				<option id="annotation_json">Json</option>
				<option>Mask</option>
			  </select>
        </div>
        <div class="col-xs-3">
			  <label for="annotation_palette">Annotation Palette:</label>
			  <select class="form-control" id="annotation_palette">
				<option>Default</option>
				<option>None</option>
			  </select>
        </div>
        </div>
        <div class="row">
        <div class="col-xs-2">
        <label>pert_color1</label> <input id="pert_color1" data-slider-id='pert_color1_C' type="text" data-slider-min="0" data-slider-max="100" data-slider-step="1" data-slider-value="20" />
        </div>
        <div class="col-xs-2">
        <label>pert_color2</label> <input id="pert_color2" data-slider-id='pert_color2_C' type="text" data-slider-min="0" data-slider-max="100" data-slider-step="1" data-slider-value="20" />
        </div>
        <div class="col-xs-2">
        <label>pert_color3</label> <input class="form-control" id="pert_color3" data-slider-id='pert_color3_C' type="text" data-slider-min="0" data-slider-max="100" data-slider-step="1" data-slider-value="20" />
        </div>
        <div class="col-xs-2">
        <label>pert_angle</label> <input id="pert_angle" data-slider-id='pert_angle_C' type="text" data-slider-min="0" data-slider-max="50" data-slider-step="1" data-slider-value="10" />
        </div>
        </div>
        <div class="row">
        <div class="col-xs-4">
        <label>pert_scale</label> <input id="pert_scale" data-slider-id='pert_scale_C' type="text" data-slider-min="0.5" data-slider-max="2" range="true" data-slider-step="0.05"  />
        </div>
        <div class="col-xs-2">
        <label>H-flip</label> <input id="pert_hflip" type="checkbox"/>
        </div>
        <div class="col-xs-2">
        <label>V-flip</label> <input id="pert_vflip" type="checkbox"/>
        </div>
        </div>
        <div class="row">
        <div class="col-xs-3">
        <label>images</label> <input id="load_images" data-slider-id='load_images_C' type="text" data-slider-min="20" data-slider-max="200" data-slider-step="10" data-slider-value="50" />
        </div>
        <div class="col-xs-3">
        <label>Perturb</label> <input id="perturb" type="checkbox" />
        </div>
        <div class="col-xs-3">
        <label>Normalize</label> <input id="normalize" type="checkbox"/>
        </div>
        <div class="col-xs-3">
		<a class="btn btn-primary btn-xs" v-on:click="reload()">Reload</a>
        </div>
        </div>
		</div>
	  </div>
	  </div>
      </div>
      <div class='row'>
          <div class="col-xs-12">
          <div class="grid">
              <div class="grid-item" height="300" width="300" v-for="image in images">
                  <img v-bind:src="image"></img>
              </div>
          </div>
          </div>
      </div>
  </section>
  <!-- /.content -->
</template>

<script>
import $ from 'jquery'
import Masonry from 'masonry-layout'
import imagesLoaded from 'imagesloaded'
import 'bootstrap-slider'

function Reload (thisData) {
  thisData.$nextTick(function () {
    var samples = $('#load_images').slider('getValue')
    $.getJSON('/api/sample?count=' + samples, function (data) {
      var pertColor1 = $('#pert_color1').slider('getValue')
      var pertColor2 = $('#pert_color2').slider('getValue')
      var pertColor3 = $('#pert_color3').slider('getValue')
      var pertAngle = $('#pert_angle').slider('getValue')
      var pertScale = $('#pert_scale').slider('getValue')
      var perturb = 0
      if ($('#perturb').prop('checked')) {
        perturb = 1
      }
      var pertHflip = 0
      if ($('#pert_hflip').prop('checked')) {
        pertHflip = 1
      }
      var pertVflip = 0
      if ($('#pert_vflip').prop('checked')) {
        pertVflip = 1
      }
      var colorspace = $('#colorspace > option:selected').text()
      var channels = 3
      var pertColorspace = colorspace
      if (colorspace === 'Grayscale') {
        channels = 1
        pertColorspace = 'BGR'
      }
      var norm = 0
      if ($('#normalize').prop('checked')) {
        norm = 1
      }
      console.log(colorspace)
      console.log(pertScale)
      var urlExtra = '&channels=' + channels +
          '&pert_color1=' + pertColor1 +
          '&pert_color2=' + pertColor2 +
          '&pert_color3=' + pertColor3 +
          '&pert_colorspace=' + pertColorspace +
          '&pert_angle=' + pertAngle +
          '&pert_min_scale=' + pertScale[0] +
          '&pert_max_scale=' + pertScale[1] +
          '&pert_hflip=' + pertHflip +
          '&pert_vflip=' + pertVflip +
          '&perturb=' + perturb +
          '&norm=' + norm
      if ($('#annotation > option:selected').text() === 'Json') {
        urlExtra += '&annotate=json'
      } else if ($('#annotation > option:selected').text() === 'Mask') {
        urlExtra += '&annotate=image&anno_factor=255'
      }
      if ($('#annotation_palette > option:selected').text() === 'Default') {
        urlExtra += '&anno_palette=default'
      } else {
        urlExtra += '&anno_palette=none'
      }

      var x = []
      for (var i in data.samples) {
        var sid = data.samples[i]
        x.push('/api/image?max_size=300&id=' + sid + urlExtra)
      }
      thisData.images = x
      thisData.$nextTick(function () {
        imagesLoaded('.grid', function () {
          var msnry = new Masonry('.grid', {
            itemSelector: '.grid-item',
            columnWidth: 200
          })
          console.log(msnry)
          msnry.on('layoutComplete', function (a, b) {
            console.log('layout done')
            console.log(b)
          })
          msnry.layout()
        })
      })
    })
  })
}

module.exports = {
  data: function () {
    return {
      images: [],
      annotation: 'None',
      use_json: false
    }
  },
  methods: {
    reload: function () { Reload(this) }
  },
  mounted: function () {
    var thisData = this
    $('#pert_color3').slider({ formatter: function (value) { return value } })
    $('#pert_color2').slider({ formatter: function (value) { return value } })
    $('#pert_color1').slider({ formatter: function (value) { return value } })
    $('#pert_angle').slider({ formatter: function (value) { return value } })
    $('#pert_scale').slider({ formatter: function (value) { return value }, value: [0.8, 1.2] })
    $('#load_images').slider({ formatter: function (value) { return value } })
    $.getJSON('/api/overview', function (overview) {
      $('#annotation_json').removeAttr('selected')
      for (var j in overview.SUMMARY) {
        var f = overview.SUMMARY[j]
        if (f.key === 'SHAPE_CNT' && f.value > 0) {
          thisData.use_json = true
          $('#annotation_json').attr('selected', true)
          console.log('use json')
          break
        }
      }
      Reload(thisData)
    })
  }
}
</script>
<style>
.info-box {
  cursor: pointer;
}
.info-box-content {
  text-align: center;
  vertical-align: middle;
  display: inherit;
}
.fullCanvas {
  width: 100%;
}
#pert_color1_C .slider-handle {
	background: blue;
}
#pert_color2_C .slider-handle {
	background: green;
}
#pert_color3_C .slider-handle {
	background: red;
}
.slider.slider-horizontal {
	width: 100px;
}
#pert_scale_C {
	width: 200px;
}

</style>
