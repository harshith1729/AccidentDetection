import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { createClient } from '@supabase/supabase-js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

const supabase = createClient(
  process.env.VITE_SUPABASE_URL,
  process.env.VITE_SUPABASE_SUPABASE_ANON_KEY
);

app.use(cors());
app.use(express.json());

app.post('/api/accident-alert', async (req, res) => {
  try {
    const { lat, lon, description, fire = 0 } = req.body;

    if (!lat || !lon || !description) {
      return res.status(400).json({
        error: 'Missing required fields: lat, lon, description'
      });
    }

    // Create the accident alert
    const { data: alert, error: alertError } = await supabase
      .from('accident_alerts')
      .insert({
        latitude: lat,
        longitude: lon,
        description
      })
      .select()
      .single();

    if (alertError) {
      console.error('Error creating alert:', alertError);
      return res.status(500).json({ error: 'Failed to create alert' });
    }

    const notifications = [];
    const notifiedDetails = [];

    if (fire === 0) {
      // Fire = 0: Find nearest 1 police and nearest 1 hospital using SQL query
      
      // Get nearest hospital
      const { data: nearestHospital } = await supabase
        .rpc('calculate_distance', { lat1: lat, lon1: lon, lat2: lat, lon2: lon })
        .select();
      
      const { data: hospitals } = await supabase
        .from('amenities')
        .select('id, name, type, latitude, longitude, email')
        .eq('type', 'hospital')
        .limit(50); // Get top 50 hospitals for distance calculation
      
      if (hospitals && hospitals.length > 0) {
        let nearest = null;
        let minDist = Infinity;
        
        for (const h of hospitals) {
          const { data: dist } = await supabase.rpc('calculate_distance', {
            lat1: h.latitude, lon1: h.longitude, lat2: lat, lon2: lon
          });
          if (dist !== null && dist < minDist) {
            minDist = dist;
            nearest = { ...h, distance: dist };
          }
        }
        
        if (nearest) {
          notifications.push({
            amenity_id: nearest.id,
            alert_id: alert.id,
            distance_km: nearest.distance
          });
          notifiedDetails.push({
            name: nearest.name,
            type: nearest.type,
            email: nearest.email,
            distance_km: nearest.distance.toFixed(2)
          });
        }
      }
      
      // Get nearest police station
      const { data: policeStations } = await supabase
        .from('amenities')
        .select('id, name, type, latitude, longitude, email')
        .eq('type', 'police')
        .limit(50);
      
      if (policeStations && policeStations.length > 0) {
        let nearest = null;
        let minDist = Infinity;
        
        for (const p of policeStations) {
          const { data: dist } = await supabase.rpc('calculate_distance', {
            lat1: p.latitude, lon1: p.longitude, lat2: lat, lon2: lon
          });
          if (dist !== null && dist < minDist) {
            minDist = dist;
            nearest = { ...p, distance: dist };
          }
        }
        
        if (nearest) {
          notifications.push({
            amenity_id: nearest.id,
            alert_id: alert.id,
            distance_km: nearest.distance
          });
          notifiedDetails.push({
            name: nearest.name,
            type: nearest.type,
            email: nearest.email,
            distance_km: nearest.distance.toFixed(2)
          });
        }
      }
      
    } else if (fire === 1) {
      // Fire = 1: Find nearest hospital, police, and fire station
      const types = ['hospital', 'police', 'fire'];
      
      for (const type of types) {
        const { data: amenities } = await supabase
          .from('amenities')
          .select('id, name, type, latitude, longitude, email')
          .eq('type', type)
          .limit(50);
        
        if (amenities && amenities.length > 0) {
          let nearest = null;
          let minDist = Infinity;
          
          for (const a of amenities) {
            const { data: dist } = await supabase.rpc('calculate_distance', {
              lat1: a.latitude, lon1: a.longitude, lat2: lat, lon2: lon
            });
            if (dist !== null && dist < minDist) {
              minDist = dist;
              nearest = { ...a, distance: dist };
            }
          }
          
          if (nearest) {
            notifications.push({
              amenity_id: nearest.id,
              alert_id: alert.id,
              distance_km: nearest.distance
            });
            notifiedDetails.push({
              name: nearest.name,
              type: nearest.type,
              email: nearest.email,
              distance_km: nearest.distance.toFixed(2)
            });
          }
        }
      }
    }

    // Insert notifications if any
    if (notifications.length > 0) {
      const { error: notifError } = await supabase
        .from('alert_notifications')
        .insert(notifications);

      if (notifError) {
        console.error('Error creating notifications:', notifError);
      }
    }

    // Format response message
    let message = `Alert created successfully!\n\n`;
    message += `Fire incident: ${fire === 1 ? 'YES' : 'NO'}\n`;
    message += `Total amenities notified: ${notifications.length}\n\n`;
    message += `📢 Alerts sent to:\n`;
    
    notifiedDetails.forEach((detail, idx) => {
      const emoji = detail.type === 'hospital' ? '🏥' : detail.type === 'police' ? '🚔' : '🚒';
      message += `${idx + 1}. ${emoji} ${detail.name}\n`;
      message += `   Type: ${detail.type}\n`;
      message += `   Email: ${detail.email}\n`;
      message += `   Distance: ${detail.distance_km} km\n\n`;
    });

    res.json({
      success: true,
      alert_id: alert.id,
      notified_amenities: notifications.length,
      fire_incident: fire === 1,
      message: message,
      details: notifiedDetails
    });

  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/amenities', async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('amenities')
      .select('id, name, type, latitude, longitude, email, address')
      .order('name');

    if (error) {
      console.error('Error fetching amenities:', error);
      return res.status(500).json({ error: 'Failed to fetch amenities' });
    }

    res.json({ amenities: data });
  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Guardian Connect API is running' });
});

app.listen(PORT, () => {
  console.log(`🚨 Guardian Connect API running on port ${PORT}`);
  console.log(`📍 POST /api/accident-alert - Create new accident alert`);
  console.log(`   - Parameters: lat, lon, description, fire (0 or 1)`);
  console.log(`   - fire=0: Notify nearest 1 police + 1 hospital`);
  console.log(`   - fire=1: Notify nearest 1 hospital + 1 police + 1 fire station`);
  console.log(`📋 GET /api/amenities - List all registered amenities`);
});